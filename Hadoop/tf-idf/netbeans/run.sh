if [ "$#" -ne 3 ]; then
    echo "Usage: ./run.sh [inputDir] [outputDir] [nTermsPerDoc]" >&2
    exit 1
fi

# HADOOP_PREFIX is path to Hadoop installation directory
# Sample: export HADOOP_PREFIX=/usr/local/haddoop/hadoop-2.4.1
# Here /usr/local/hadoop/hadoop is a symbolic link to Hadoop Install
export HADOOP_PREFIX=/usr/local/hadoop/hadoop

# JAVA_HOME is path to JDK installation directory
export JAVA_HOME=/usr/lib/jvm/java

# Build CLASSPATH to add all Hadoop related Jars
export CLASSPATH=$HADOOP_PREFIX/etc/hadoop
BASE_HADOOP_JAR_DIR=$HADOOP_PREFIX/share/hadoop
for f in \
   $BASE_HADOOP_JAR_DIR/common/*.jar   \
   $BASE_HADOOP_JAR_DIR/common/lib/*.jar  \
   $BASE_HADOOP_JAR_DIR/hdfs/*.jar \
   $BASE_HADOOP_JAR_DIR/hdfs/lib/*.jar \
   $BASE_HADOOP_JAR_DIR/yarn/*.jar \
   $BASE_HADOOP_JAR_DIR/yarn/lib/*.jar \
   $BASE_HADOOP_JAR_DIR/mapreduce/*.jar \
   $BASE_HADOOP_JAR_DIR/mapreduce/lib/*.jar
do
  CLASSPATH=$CLASSPATH:$f
done

CLASSPATH=$CLASSPATH:dist/bdpuh.project.jar

java -classpath ${CLASSPATH} bdpuh.project.tf_idf $1 $2 $3
