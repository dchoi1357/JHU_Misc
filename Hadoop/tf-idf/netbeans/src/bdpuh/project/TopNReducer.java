package bdpuh.project;

import java.io.IOException;
import java.util.Arrays;
import java.util.PriorityQueue;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

class TopNReducer extends Reducer<Text, Text, Text, Text> {
	int topNterms;
	double tf_idf;
	PriorityQueue<term_tfidf> pq;
	Text outTerms = new Text();

	private static class term_tfidf implements Comparable<term_tfidf> {
		String term;
		double tf_idf;

		public term_tfidf(String inTerm, String tfidf_string) {
			term = inTerm;
			tf_idf = Double.parseDouble(tfidf_string);
		}

		@Override
		public String toString() { return term + ": " + tf_idf; }

		@Override
		public int compareTo(term_tfidf other) {
			return Double.compare(this.tf_idf, other.tf_idf);
		}
	}

	@Override
	protected void setup(Context context)
			throws IOException, InterruptedException {
		super.setup(context);
		topNterms = context.getConfiguration().getInt("topN", 0);
	}

	@Override
	protected void reduce(Text docName, Iterable<Text> values,
			Context context) throws IOException, InterruptedException {
		pq = new PriorityQueue<>(); // priority queue for keeping term + tf-idf

		for (Text t : values) { // loop through all terms within one doc
			String[] toks = t.toString().split("\\t"); // term tf-idf
			pq.add(new term_tfidf(toks[0], toks[1]));
			if (pq.size() > topNterms) { // remove element if size > topNterm
				pq.poll();
			}
		}

		String[] topTerms = new String[topNterms];
		for (int n = topNterms; n > 0; n--) {
			topTerms[n - 1] = pq.remove().term;
		}
		outTerms.set( Arrays.toString(topTerms) );
		context.write(docName, outTerms);
	}
}
