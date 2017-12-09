package bdpuh.project;

import java.io.IOException;
import java.util.Random;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;

public class tf_idf {
	static Random r = new Random();
	static String cdir = "/tf-idf";
	static Configuration cfg;
	static int nInputs, nReducer, topN;

	/**
	 * @param args the command line arguments
	 * @throws java.io.IOException
	 * @throws java.lang.InterruptedException
	 * @throws java.lang.ClassNotFoundException
	 */
	public static void main(String[] args) throws IOException,
			InterruptedException, ClassNotFoundException {
		Path inPath, outPath;
		cfg = new Configuration(); // Hadoop config
		
		if (args.length != 3) {
			return; // incorrect argument number
		} else {
			inPath = new Path(cdir, args[0]);
			outPath = new Path(cdir, args[1]);
			topN = Integer.parseInt(args[2]);
		}
		FileSystem fs = FileSystem.newInstance(cfg);
		nInputs = fs.listStatus(inPath).length; // number of input files
		cfg.setInt("nDocs", nInputs); // pass no. of total input docs via cfg
		nReducer = nInputs / 10; // each reducer gets 1/10th of share
		cfg.setInt("topN", topN); // top N terms to pick for each doc
		
		Path[] tmpDs = new Path[3];
		for (int n=0; n<tmpDs.length; n++) {
			tmpDs[n] = new Path(cdir, "t_" + Integer.toHexString(r.nextInt()));
		}
		
//		cfg.set("mapreduce.framework.name", "local");
		termCount(inPath, tmpDs[0]); // count word freq per doc
		termFreq(tmpDs[0], tmpDs[1]); // count word freq across all docs
		idf(tmpDs[1], tmpDs[2]); // combine previous 2, calculate tf-idf
		takeTopN(tmpDs[2], outPath); // pick top N per doc
		
		for (Path d: tmpDs) { fs.delete(d, true); } // delete tmpDirs
		fs.close();
	}

	private static void termCount(Path in, Path out) throws IOException,
			InterruptedException, ClassNotFoundException {
		Job jb = genSkeletonJob(in, out, nReducer, "Term Count per Doc",
				TextInputFormat.class, SequenceFileOutputFormat.class);

		jb.setMapOutputValueClass(IntWritable.class);
		jb.setMapperClass(TermCountMapper.class); // Map/Reduce classes
		jb.setReducerClass(TermCountReducer.class);
		jb.waitForCompletion(true); // start run
	}

	private static void termFreq(Path in, Path out) throws IOException,
			InterruptedException, ClassNotFoundException {
		Job jb = genSkeletonJob(in, out, nReducer, "Term Freq",
				SequenceFileInputFormat.class, SequenceFileOutputFormat.class);

		jb.setMapperClass(FirstKeyMapper.class); // Map/Reduce classes
		jb.setReducerClass(TermFreqReducer.class);
		jb.waitForCompletion(true); // start run
	}

	private static void idf(Path in, Path out) throws IOException,
			InterruptedException, ClassNotFoundException {
		Job jb = genSkeletonJob(in, out, nReducer, "Inverse Doc Freq",
				SequenceFileInputFormat.class, SequenceFileOutputFormat.class);

		jb.setMapperClass(FirstKeyMapper.class); // Map/Reduce classes
		jb.setReducerClass(idfReducer.class);
		jb.waitForCompletion(true); // start run
	}

	private static void takeTopN(Path in, Path out) throws IOException,
			InterruptedException, ClassNotFoundException {
		Job jb = genSkeletonJob(in, out, 1, "Top N Terms",
				SequenceFileInputFormat.class, TextOutputFormat.class);

		jb.setMapperClass(FirstKeyMapper.class); // Map/Reduce classes
		jb.setReducerClass(TopNReducer.class);
//		jb.setCombinerClass(TopNcombiner.class);
		jb.waitForCompletion(true); // start run
	}
	
	private static Job genSkeletonJob(Path in, Path out, int n, String title,
			Class inFormat, Class outFormat) 
			throws IOException {
		Job jb = new Job(cfg, title);
		FileInputFormat.addInputPath(jb, in);
		FileOutputFormat.setOutputPath(jb, out);
		
		jb.setMapOutputKeyClass(Text.class);
		jb.setOutputKeyClass(Text.class);
		jb.setOutputValueClass(Text.class);
		jb.setInputFormatClass(inFormat); // I/O classes
		jb.setOutputFormatClass(outFormat);
		
		jb.setNumReduceTasks(n);
		jb.setJarByClass(bdpuh.project.tf_idf.class);		
		return jb;
	}
}
