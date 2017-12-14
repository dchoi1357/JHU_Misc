package bdpuh.project;

import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

class idfReducer extends Reducer<Text, Text, Text, Text> {
	Text outKey = new Text();
	Text tf_idf = new Text();
	double nDocsInCorpus;
	LinkedList<String> docTerms;
	LinkedList<Double> termFreq;
	
	@Override
	protected void setup(Context context) 
			throws IOException, InterruptedException {
		super.setup(context);
		nDocsInCorpus = (double) context.getConfiguration().getInt("nDocs", 0);
	}
	
	@Override
	protected void reduce(Text term, Iterable<Text> values, 
			Context context) throws IOException, InterruptedException {
		docTerms = new LinkedList<>();
		termFreq = new LinkedList<>();
		int nAppearence = 0;

		for (Text t: values) { 
			nAppearence++; // increment number of docs term appear in
			
			String[] toks = t.toString().split("\\t"); // fName freq wordCt
			docTerms.add(toks[0] + "\t" + term.toString());
			
			int freq = Integer.parseInt(toks[1]);
			int wordCt = Integer.parseInt(toks[2]);
			termFreq.add((double) freq / wordCt);
		}
		
		// Loop over cached values and write out results
		Iterator<String> dt = docTerms.iterator();
		Iterator<Double> tf = termFreq.iterator();
		double idf = Math.log(nDocsInCorpus / nAppearence); // idf for term
		while (dt.hasNext() && tf.hasNext()) {
			outKey.set(dt.next());
			tf_idf.set(Double.toString(tf.next() * idf));
			context.write(outKey, tf_idf);
		}
	}
}
