import com.google.common.base.Optional;
import com.google.common.base.Splitter;
import com.google.common.collect.FluentIterable;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.util.List;


public final class KMeansMP {
	private static final Splitter COMMA_SPLITTER = Splitter.on(',').trimResults().omitEmptyStrings();
	private static final String UNKNOWN_CAR_MAKER = "UnknownCarMaker";

	private static class ExtractConsumption implements Function<String, Vector>{
		@Override
		public Vector call(String line) throws Exception {
			List<String> items = COMMA_SPLITTER.splitToList(line);
			double[] consumption = new double[items.size()];
			for (int i = 1; i < items.size(); i++){
				consumption[i-1] = Double.valueOf(items.get(i));
			}
			return Vectors.dense(consumption);
		}
	}

	private static class ExtractCarMaker implements Function<String, String>{
		@Override
		public String call(String line) throws Exception {
			Optional<String> carMaker = FluentIterable.from(COMMA_SPLITTER.split(line)).first();
			if (carMaker.isPresent()){
				return carMaker.get();
			}else{
				return UNKNOWN_CAR_MAKER;
			}
		}
	}

	private static class CarCluster implements PairFunction<Tuple2<String, Vector>, Integer, String>{
		private KMeansModel model;
		public CarCluster(KMeansModel model){
			this.model = model;
		}

		@Override
		public Tuple2<Integer, String> call(Tuple2<String, Vector> cluster) throws Exception {
			return new Tuple2<>(model.predict(cluster._2()), cluster._1());
		}
	}

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println(
                    "Usage: KMeansMP <input_file> <results>");
            System.exit(1);
        }
        String inputFile = args[0];
        String results_path = args[1];
        JavaPairRDD<Integer, Iterable<String>> results;
        int k = 4;
        int iterations = 100;
        int runs = 1;
        long seed = 0;
		final KMeansModel model;
		
        SparkConf sparkConf = new SparkConf().setAppName("KMeans MP");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

		JavaRDD<String> lines = sc.textFile(inputFile);
		JavaRDD<Vector> consumption = lines.map(new ExtractConsumption());
		JavaRDD<String> carMakers = lines.map(new ExtractCarMaker());
		model = KMeans.train(consumption.rdd(), k, iterations, runs, KMeans.RANDOM(), seed);
		results = carMakers.zip(consumption).mapToPair(new CarCluster(model)).groupByKey();
		results.saveAsTextFile(results_path);
        sc.stop();
    }
}