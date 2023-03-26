package org.apache.spark.ml.made

import breeze.stats.distributions.Gaussian
import breeze.{linalg => l}
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}


trait LinearRegressionParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  private val maxIterations = new IntParam(this, "maxIterations", "Maximum number of iterations in gradient descent loop")
  private val learningRate = new DoubleParam(
    this, "learningRate", "Determines the step size at each iteration while moving toward a minimum of a loss function")


  def setLearningRate(value: Double): this.type = set(learningRate, value)

  def setMaxIterations(value: Int): this.type = set(maxIterations, value)

  def getLearningRate: Double = $(learningRate)

  def getMaxIterations: Int = $(maxIterations)

  setDefault(maxIterations -> 1000)
  setDefault(learningRate -> 1.0)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array($(inputCol), "ones", $(outputCol)))
      .setOutputCol("features_target")

    val vectors: Dataset[Vector] = assembler
      .transform(dataset.withColumn("ones", lit(1)))
      .select("features_target")
      .as[Vector]

    val dim: Int = AttributeGroup.fromStructField(dataset.schema($(inputCol))).numAttributes.getOrElse(
      vectors.first().size
    )

    var weights: l.DenseVector[Double] = l.DenseVector.rand[Double](dim - 1, Gaussian(0, 1))

    for (_ <- 0 until getMaxIterations) {
      val summary = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer= new MultivariateOnlineSummarizer()
        data.foreach(v => {
          val X = v.asBreeze(0 until weights.size).toDenseVector
          val y = v.asBreeze(weights.size)
          val grad = X * (l.sum(X * weights) - y)
          summarizer.add(mllib.linalg.Vectors.fromBreeze(grad))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)

      weights = weights - getLearningRate * summary.mean.asBreeze
    }

    copyValues(new LinearRegressionModel(Vectors.fromBreeze(weights))).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]


class LinearRegressionModel private[made](override val uid: String,
                                          val weights: Vector) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {
  private[made] def this(weights: Vector) =
    this(Identifiable.randomUID("LinRegModel"), weights)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(weights), extra)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      sqlContext.createDataFrame(Seq(Tuple1(weights))).write.parquet(path + "/weights")
    }
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val bias = weights.asBreeze(-1)
    val weightsBreeze = weights.asBreeze(0 until weights.size - 1)

    val transformUdf = {
      dataset.sqlContext.udf.register(uid + "_transform",
        (x: Vector) => bias + x.asBreeze.dot(weightsBreeze)
      )
    }

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)
      val vectors = sqlContext.read.parquet(path + "/weights")
      implicit val encoder: Encoder[Vector] = ExpressionEncoder()
      val weights = vectors.select(vectors("_1").as[Vector]).first()

      val model = new LinearRegressionModel(weights)
      metadata.getAndSetParams(model)
      model
    }
  }
}
