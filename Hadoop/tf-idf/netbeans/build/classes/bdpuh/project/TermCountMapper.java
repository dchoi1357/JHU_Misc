package bdpuh.project;

import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

public class TermCountMapper extends Mapper<LongWritable, Text, 
		Text, IntWritable> {
	String fileInName;
	IntWritable ONE = new IntWritable(1);
	Text outKey = new Text();
	
	@Override
	protected void setup(Context context) 
			throws IOException, InterruptedException {
		super.setup(context);
		fileInName = ((FileSplit) context.getInputSplit()).getPath().getName();
	}
	
	@Override
	protected void map(LongWritable key, Text value, Context context) 
			throws IOException, InterruptedException {
		String[] toks = value.toString().split("[,.\\s]+", 0); // split by \s,.
		for (String tok : toks) {
			outKey.set(fileInName + "\t" + tok); // filename and word
			context.write(outKey, ONE);
		}
	}
}
