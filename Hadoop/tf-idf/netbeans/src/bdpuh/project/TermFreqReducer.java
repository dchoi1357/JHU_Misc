package bdpuh.project;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedList;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

class TermFreqReducer extends Reducer <Text, Text, Text, Text> {
	LinkedList<String> terms;
	LinkedList<Integer> counts;
	Text outVal = new Text();
	Text termAndFile = new Text();
	
	@Override
	protected void reduce(Text key, Iterable<Text> values, 
			Context context) throws IOException, InterruptedException {
		int docWordCount = 0, tmp; // total words in this doc
		String term;
		
		terms = new LinkedList<>();
		counts = new LinkedList<>();
		for (Text v : values) {
			String[] t = v.toString().split("\\t", 0);
			tmp = Integer.parseInt(t[1]); // count for term in this doc 
			docWordCount += tmp; // sum up total word count for doc
			
			terms.add(t[0]); // the current word
			counts.add(tmp); // count for current word in doc
		}
		
		Iterator<String> t = terms.iterator();
		Iterator<Integer> c = counts.iterator();
		while (t.hasNext() && c.hasNext()) {
			termAndFile.set(t.next() + "\t" + key); // term \t fileName
			outVal.set("" + c.next() + "\t" + docWordCount);// freq \t docLength
			context.write(termAndFile, outVal);
		}
	}
}
