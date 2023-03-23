package org.apache.spark.ml.made

import breeze.{linalg => l}
import org.apache.spark.ml.attribute.AttributeGroup
import breeze.stats.distributions.Gaussian
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param, ParamMap}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row, SparkSession}


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
  setDefault(learningRate -> 0.01)

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

  private def gradStep(data: Vector, weights: l.DenseVector[Double], bias: Double):
  (l.DenseVector[Double], Double) = {
    val features = data.asBreeze(0 to data.size)
    val target = data.asBreeze(-1)

    val error = target - ((features dot weights) + bias)
    val weightsGrad = -2.0 * features.toDenseVector * error
    val biasGrad = -2.0 * bias
    (weightsGrad, biasGrad)
  }

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()
    val vectors: Dataset[Vector] = dataset.select(dataset($(inputCol)).as[Vector])

    val dim: Int = AttributeGroup.fromStructField((dataset.schema($(inputCol)))).numAttributes.getOrElse(
      vectors.first().size
    )

    val weights: l.DenseVector[Double] = l.DenseVector.rand[Double](dim - 1, Gaussian(0, 1))
    var bias: Double = 0

    for (_ <- 0 to getMaxIterations) {
      val vectorsRdd = vectors.rdd
      val weightsRdd = vectorsRdd.sparkContext.broadcast(weights)

      val (weightsGrad, biasGrad) = vectorsRdd
        .map(data => gradStep(data, weightsRdd.value, bias))
        .reduce((gradsL, gradsR) => (gradsL._1 + gradsR._1, gradsL._2 + gradsR._2))

      weights -= weightsGrad * (getLearningRate / vectorsRdd.count())
      bias -= biasGrad * (getLearningRate / vectorsRdd.count())
    }
    val result = l.DenseVector.vertcat(weights, l.DenseVector[Double](bias))
    copyValues(new LinearRegressionModel(Vectors.fromBreeze(result))).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}


class LinearRegressionModel private[made](override val uid: String,
                                          val weights: Vector) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {
  private[made] def this(weights: Vector) =
    this(Identifiable.randomUID("LinRegModel"), weights)

  override def copy(extra: ParamMap): LinearRegressionModel = ???

  override def write: MLWriter = ???

  override def transform(dataset: Dataset[_]): DataFrame = ???

  override def transformSchema(schema: StructType): StructType = ???
}