package bdpuh.project;

import java.io.IOException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

class FirstKeyMapper extends Mapper <Text, Text, Text, Text> {
	Text outKey = new Text();
	Text outVal = new Text();
	
	@Override
	protected void map(Text key, Text value, Context context)
			throws IOException, InterruptedException {
		String[] toks =  key.toString().split("\\t", 2); //split by tab once
		
		outKey.set(toks[0]); // substring before first tab
		if (toks.length > 1) { 
			outVal.set(toks[1] + "\t" + value); // rest of key after first tab
		} else {
			outVal.set(value);
		}
		
		context.write(outKey, outVal);
	}
}