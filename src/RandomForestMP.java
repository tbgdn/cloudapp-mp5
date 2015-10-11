import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import java.util.HashMap;

public final class RandomForestMP {

	private static class ExtractFeatures implements Function<String, LabeledPoint>{
		@Override
		public LabeledPoint call(String line) throws Exception {
			String[] points = line.split(",");
			double[] features = new double[points.length-1];
			for (int i = 0; i < points.length - 1; i++){
				features[i] = Double.valueOf(points[i]);
			}
			return new LabeledPoint(Double.valueOf(points[points.length-1]), Vectors.dense(features));
		}
	}

	private static class ExtractPredictions implements Function<LabeledPoint, LabeledPoint>{
		RandomForestModel model;
		public ExtractPredictions(RandomForestModel model){
			this.model = model;
		}

		@Override
		public LabeledPoint call(LabeledPoint input) throws Exception {
			Vectors.dense(input.label(), input.features().toArray());
			return new LabeledPoint(model.predict(input.features()),
							Vectors.dense(input.label(), input.features().toArray()));
		}
	}

    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println(
                    "Usage: RandomForestMP <training_data> <test_data> <results>");
            System.exit(1);
        }
        String training_data_path = args[0];
        String test_data_path = args[1];
        String results_path = args[2];

        SparkConf sparkConf = new SparkConf().setAppName("RandomForestMP");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        final RandomForestModel model;

        Integer numClasses = 2;
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        Integer numTrees = 3;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        Integer maxDepth = 5;
        Integer maxBins = 32;
        Integer seed = 12345;

		// TODO
		JavaRDD<LabeledPoint> trainData = sc.textFile(training_data_path).map(new ExtractFeatures());
		JavaRDD<LabeledPoint> test = sc.textFile(test_data_path).map(new ExtractFeatures());
		model = RandomForest.trainClassifier(trainData, numClasses, categoricalFeaturesInfo, numTrees,
					 featureSubsetStrategy, impurity, maxDepth, maxBins, seed);

        JavaRDD<LabeledPoint> results = test.map(new ExtractPredictions(model));
        results.saveAsTextFile(results_path);
        sc.stop();
    }

}
