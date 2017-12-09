package bdpuh.project;

import java.io.IOException;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

class FirstKeyMapper extends Mapper <LongWritable, Text, Text, Text> {
	Text outKey = new Text();
	Text outVal = new Text();
	
	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		String[] toks =  value.toString().split("\\t", 2); //split by tab once
		
		outKey.set(toks[0]); // substring before first tab
		outVal.set(toks[1]); // rest of string after first tab
		context.write(outKey, outVal);
	}
}