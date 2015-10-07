import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.edge.Edge;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;

import java.io.IOException;

/**
 * Implementation of the connected component algorithm that identifies
 * connected components and assigns each vertex its "component
 * identifier" (the smallest vertex id in the component).
 */
public class ConnectedComponentsComputation extends
    BasicComputation<IntWritable, IntWritable, NullWritable, IntWritable> {
  /**
   * Propagates the smallest vertex id to all neighbors. Will always choose to
   * halt and only reactivate if a smaller id has been sent to it.
   *
   * @param vertex Vertex
   * @param messages Iterator of messages from the previous superstep.
   * @throws IOException
   */
  @Override
  public void compute(Vertex<IntWritable, IntWritable, NullWritable> vertex,
					  Iterable<IntWritable> messages) throws IOException {
	  if (getSuperstep() == 0){
		  computeSuperStep(vertex);
	  }else{
		  computeRegularStep(vertex, messages);
	  }
	  vertex.voteToHalt();
  }

	private void computeSuperStep(Vertex<IntWritable, IntWritable, NullWritable> vertex) {
		int minVertex = findMinVertex(vertex);
		if (vertex.getValue().get() != minVertex){
			vertex.setValue(new IntWritable(minVertex));
			propagateVertex(vertex);
		}
	}

	private int findMinVertex(Vertex<IntWritable, IntWritable, NullWritable> vertex) {
		int minVertex = vertex.getValue().get();
		for(Edge<IntWritable, NullWritable> edge: vertex.getEdges()){
			int neighbour = edge.getTargetVertexId().get();
			if (neighbour < minVertex){
				minVertex = neighbour;
			}
		}
		return minVertex;
	}

	private void propagateVertex(Vertex<IntWritable, IntWritable, NullWritable> vertex) {
		for(Edge<IntWritable, NullWritable> edge: vertex.getEdges()){
			int neighbour = edge.getTargetVertexId().get();
			if (vertex.getValue().get() < neighbour){
				sendMessage(edge.getTargetVertexId(), vertex.getValue());
			}
		}
	}

	private void computeRegularStep(Vertex<IntWritable, IntWritable, NullWritable> vertex,
									Iterable<IntWritable> messages) {
		int minId = findMinId(messages);
		if (minId < vertex.getValue().get()){
			vertex.setValue(new IntWritable(minId));
			sendMessageToAllEdges(vertex, vertex.getValue());
		}
	}

	private int findMinId(Iterable<IntWritable> collection){
		int min = Integer.MAX_VALUE;
		for (IntWritable item: collection){
			if (item.get() < min){
				min = item.get();
			}
		}
		return min;
	}
}
