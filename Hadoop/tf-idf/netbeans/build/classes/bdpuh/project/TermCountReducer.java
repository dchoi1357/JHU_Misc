package bdpuh.project;

import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

class TermCountReducer extends Reducer<Text, IntWritable, Text, Text> {
	Text count = new Text();
	
	@Override
	protected void reduce(Text key, Iterable<IntWritable> values, 
			Context context) throws IOException, InterruptedException {
		int sum = 0;
		for (IntWritable v : values) {
			sum += v.get();
		}
		count.set(Integer.toString(sum));
		context.write(key, count);
	}
}
