"""
Apache Spark Structured Streaming Pipeline for Diabetes Data
Simulates real-time data ingestion and processing
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import time
import os
import shutil

class SparkStreamingPipeline:
    def __init__(self, institution_name, data_path, checkpoint_path):
        self.institution_name = institution_name
        self.data_path = data_path
        self.checkpoint_path = checkpoint_path
        
        # Initialize Spark Session
        self.spark = SparkSession.builder \
            .appName(f"DiabetesStreaming_{institution_name}") \
            .master("local[*]") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        print(f"Spark Session initialized for {institution_name}")
    
    def define_schema(self):
        """Define schema for incoming diabetes data"""
        schema = StructType([
            StructField("timestamp", StringType(), True),
            StructField("glucose", DoubleType(), True),
            StructField("glucose_ma_1h", DoubleType(), True),
            StructField("glucose_std_1h", DoubleType(), True),
            StructField("glucose_roc", DoubleType(), True),
            StructField("steps", IntegerType(), True),
            StructField("heart_rate", IntegerType(), True),
            StructField("hour", IntegerType(), True),
            StructField("day_of_week", IntegerType(), True),
            StructField("hypo_risk", IntegerType(), True)
        ])
        return schema
    
    def create_streaming_source(self):
        """
        Create streaming DataFrame from JSON files
        Simulates real-time data ingestion
        """
        schema = self.define_schema()
        
        # Read streaming data
        stream_df = self.spark \
            .readStream \
            .schema(schema) \
            .option("maxFilesPerTrigger", 1) \
            .json(self.data_path)
        
        print(f"Streaming source created from {self.data_path}")
        return stream_df
    
    def preprocess_stream(self, stream_df):
        """
        Apply preprocessing and feature engineering on streaming data
        """
        print("Applying preprocessing transformations...")
        
        # Convert timestamp to proper format
        processed_df = stream_df.withColumn(
            "timestamp", 
            to_timestamp(col("timestamp"))
        )
        
        # Handle missing values
        processed_df = processed_df.fillna({
            'glucose': 120.0,
            'glucose_ma_1h': 120.0,
            'glucose_std_1h': 0.0,
            'glucose_roc': 0.0,
            'steps': 0,
            'heart_rate': 75
        })
        
        # Add derived features
        processed_df = processed_df.withColumn(
            "is_night", 
            when((col("hour") >= 22) | (col("hour") <= 6), 1).otherwise(0)
        )
        
        processed_df = processed_df.withColumn(
            "is_weekend",
            when(col("day_of_week").isin([5, 6]), 1).otherwise(0)
        )
        
        # Glucose categories
        processed_df = processed_df.withColumn(
            "glucose_category",
            when(col("glucose") < 70, "low")
            .when(col("glucose") > 180, "high")
            .otherwise("normal")
        )
        
        # Add processing timestamp
        processed_df = processed_df.withColumn(
            "processing_time",
            current_timestamp()
        )
        
        return processed_df
    
    def create_features(self, df):
        """
        Create feature vector for machine learning
        """
        feature_columns = [
            'glucose', 'glucose_ma_1h', 'glucose_std_1h', 'glucose_roc',
            'steps', 'heart_rate', 'hour', 'day_of_week', 'is_night', 'is_weekend'
        ]
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="features_raw"
        )
        
        # Scale features
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[assembler, scaler])
        
        return pipeline, feature_columns
    
    def write_to_memory(self, processed_df, query_name):
        """
        Write streaming data to memory table for querying
        """
        query = processed_df \
            .writeStream \
            .outputMode("append") \
            .format("memory") \
            .queryName(query_name) \
            .option("checkpointLocation", f"{self.checkpoint_path}/{query_name}") \
            .start()
        
        print(f"Started memory sink: {query_name}")
        return query
    
    def write_to_console(self, processed_df, query_name):
        """
        Write streaming data to console for debugging
        """
        query = processed_df \
            .writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", False) \
            .option("numRows", 10) \
            .option("checkpointLocation", f"{self.checkpoint_path}/{query_name}") \
            .start()
        
        print(f"Started console sink: {query_name}")
        return query
    
    def write_to_parquet(self, processed_df, output_path, query_name):
        """
        Write streaming data to Parquet files
        """
        query = processed_df \
            .writeStream \
            .outputMode("append") \
            .format("parquet") \
            .option("path", output_path) \
            .option("checkpointLocation", f"{self.checkpoint_path}/{query_name}") \
            .partitionBy("glucose_category") \
            .start()
        
        print(f"Started parquet sink: {query_name}")
        return query
    
    def compute_statistics(self, query_name):
        """
        Compute real-time statistics from memory table
        """
        stats_df = self.spark.sql(f"""
            SELECT 
                COUNT(*) as total_records,
                AVG(glucose) as avg_glucose,
                MIN(glucose) as min_glucose,
                MAX(glucose) as max_glucose,
                STDDEV(glucose) as std_glucose,
                SUM(CASE WHEN hypo_risk = 1 THEN 1 ELSE 0 END) as hypo_events,
                AVG(heart_rate) as avg_heart_rate,
                SUM(steps) as total_steps
            FROM {query_name}
        """)
        
        return stats_df
    
    def run_pipeline(self, duration_seconds=60, write_output=True):
        """
        Run the complete streaming pipeline
        """
        print("="*60)
        print(f"Starting Streaming Pipeline for {self.institution_name}")
        print("="*60)
        
        # Create streaming source
        stream_df = self.create_streaming_source()
        
        # Preprocess stream
        processed_df = self.preprocess_stream(stream_df)
        
        # Start memory sink for querying
        query_name = f"{self.institution_name}_stream"
        memory_query = self.write_to_memory(processed_df, query_name)
        
        # Optional: Start console sink for debugging
        # console_query = self.write_to_console(processed_df.select(
        #     "timestamp", "glucose", "glucose_category", "hypo_risk"
        # ), f"{query_name}_console")
        
        # Optional: Write to Parquet
        if write_output:
            output_path = f"data/streams/{self.institution_name}_output"
            parquet_query = self.write_to_parquet(
                processed_df, 
                output_path, 
                f"{query_name}_parquet"
            )
        
        print(f"\nPipeline running... (Duration: {duration_seconds}s)")
        print("Processing streaming data...\n")
        
        # Run for specified duration
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            time.sleep(5)
            
            # Compute and display statistics
            try:
                stats = self.compute_statistics(query_name)
                stats_data = stats.collect()[0]
                
                print(f"\n[{self.institution_name}] Statistics:")
                print(f"  Total Records: {stats_data['total_records']}")
                print(f"  Avg Glucose: {stats_data['avg_glucose']:.2f} mg/dL")
                print(f"  Glucose Range: [{stats_data['min_glucose']:.1f}, {stats_data['max_glucose']:.1f}]")
                print(f"  Hypo Events: {stats_data['hypo_events']}")
                print(f"  Avg Heart Rate: {stats_data['avg_heart_rate']:.1f} bpm")
            except Exception as e:
                print(f"Waiting for data... ({int(time.time() - start_time)}s)")
        
        # Get final data for ML
        print("\nExtracting processed data for ML training...")
        final_df = self.spark.sql(f"SELECT * FROM {query_name}")
        
        # Stop all queries
        print("\nStopping streaming queries...")
        for query in self.spark.streams.active:
            query.stop()
        
        print("="*60)
        print(f"Pipeline completed for {self.institution_name}")
        print("="*60)
        
        return final_df
    
    def prepare_ml_data(self, df):
        """
        Prepare data for machine learning
        Returns features and labels as Pandas DataFrame
        """
        print("Preparing ML data...")
        
        # Create feature vector
        pipeline, feature_columns = self.create_features(df)
        pipeline_model = pipeline.fit(df)
        df_features = pipeline_model.transform(df)
        
        # Select relevant columns
        ml_df = df_features.select(
            "features",
            "hypo_risk"
        )
        
        # Convert to Pandas
        pandas_df = ml_df.toPandas()
        
        print(f"ML data prepared: {len(pandas_df)} records")
        return pandas_df, feature_columns
    
    def stop(self):
        """Stop Spark session"""
        self.spark.stop()
        print(f"Spark session stopped for {self.institution_name}")


def simulate_streaming_data(source_file, stream_dir, delay_ms=100):
    """
    Simulate streaming by copying data files gradually
    This simulates real-time data arrival
    """
    import json
    
    os.makedirs(stream_dir, exist_ok=True)
    
    # Read source data
    with open(source_file, 'r') as f:
        records = [json.loads(line) for line in f]
    
    print(f"Simulating stream from {source_file}")
    print(f"Total records: {len(records)}")
    
    # Write records in batches
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        batch_file = os.path.join(stream_dir, f"batch_{i//batch_size}.json")
        
        with open(batch_file, 'w') as f:
            for record in batch:
                f.write(json.dumps(record) + '\n')
        
        print(f"Written batch {i//batch_size + 1} ({len(batch)} records)")
        time.sleep(delay_ms / 1000.0)


if __name__ == "__main__":
    import sys
    import threading
    
    # Configuration
    INSTITUTION_NAME = "institution_1"
    SOURCE_FILE = f"data/processed/{INSTITUTION_NAME}_stream.jsonl"
    STREAM_DIR = f"data/streams/{INSTITUTION_NAME}_input"
    CHECKPOINT_PATH = f"data/checkpoints/{INSTITUTION_NAME}"
    
    # Clean up previous runs
    for path in [STREAM_DIR, CHECKPOINT_PATH]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    
    # Start data simulation in background thread
    print("Starting data streaming simulation...")
    stream_thread = threading.Thread(
        target=simulate_streaming_data,
        args=(SOURCE_FILE, STREAM_DIR, 500)
    )
    stream_thread.daemon = True
    stream_thread.start()
    
    # Wait for some data to be available
    time.sleep(2)
    
    # Initialize and run pipeline
    pipeline = SparkStreamingPipeline(
        institution_name=INSTITUTION_NAME,
        data_path=STREAM_DIR,
        checkpoint_path=CHECKPOINT_PATH
    )
    
    try:
        # Run pipeline
        final_df = pipeline.run_pipeline(duration_seconds=30, write_output=True)
        
        # Prepare data for ML
        ml_data, feature_cols = pipeline.prepare_ml_data(final_df)
        
        # Save ML data
        output_file = f"data/processed/{INSTITUTION_NAME}_ml_data.parquet"
        ml_data.to_parquet(output_file, index=False)
        print(f"\nML data saved to {output_file}")
        
    finally:
        pipeline.stop()