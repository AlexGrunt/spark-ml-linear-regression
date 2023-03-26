package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix, DenseVector}
import com.google.common.io.Files
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.scalatest.Assertion
import org.scalatest.flatspec._
import org.scalatest.matchers._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta = 0.01

  lazy val df: DataFrame = LinearRegressionTest._df
  lazy val weights: DenseVector[Double] = LinearRegressionTest._weights
  lazy val bias: Double = LinearRegressionTest._bias
  lazy val y: DenseVector[Double] = LinearRegressionTest._y


  "Model" should "evaluate predictions" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = Vectors.fromBreeze(
        DenseVector.vertcat(weights, DenseVector[Double](bias))
      )
    ).setInputCol("features").setOutputCol("label")

    validateModel(model, model.transform(df))
  }

  "Estimator" should "produce functional model" in {
    val estimator = new LinearRegression().setInputCol("features").setOutputCol("label")
    val model = estimator.fit(df)

    model.weights.size should be(weights.size + 1)
    model.weights(0) should be(weights(0) +- delta)
    model.weights(1) should be(weights(1) +- delta)
    model.weights(2) should be(weights(2) +- delta)
    model.weights(3) should be(bias +- delta)

    validateModel(model, model.transform(df))
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("label")
    ))

    val tempDir = Files.createTempDir()

    pipeline.write.overwrite().save(tempDir.getAbsolutePath)

    val model = Pipeline.load(tempDir.getAbsolutePath)
      .fit(df)
      .stages(0)
      .asInstanceOf[LinearRegressionModel]

    model.weights should be(weights.size + 1)
    model.weights(0) should be(weights(0) +- delta)
    model.weights(1) should be(weights(0) +- delta)
    model.weights(2) should be(weights(0) +- delta)
    model.weights(3) should be(bias +- delta)
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("label")
    ))
    val model = pipeline.fit(df)
    val tempDir = Files.createTempDir()

    model.write.overwrite().save(tempDir.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tempDir.getAbsolutePath)
    validateModel(model.stages(0).asInstanceOf[LinearRegressionModel], reRead.transform(df))
  }

  private def validateModel(model: LinearRegressionModel, df: DataFrame): Unit = {
    val vectors: Array[Double] = df.collect().map(_.getAs[Double](1))

    vectors.length should be(100000)
    for (i <- vectors.indices)
      vectors(i) should be(y(i) +- delta)
  }
}

object LinearRegressionTest extends WithSpark {

  import sqlc.implicits._

  lazy val _X: DenseMatrix[Double] = DenseMatrix.rand[Double](100000, 3)
  lazy val _weights: DenseVector[Double] = DenseVector(1.5, 0.3, -0.7)
  lazy val _bias: Double = 1.0
  lazy val _y: DenseVector[Double] = _X * _weights + _bias + DenseVector.rand(100000) * 0.0001

  lazy val data: DenseMatrix[Double] = DenseMatrix.horzcat(_X, _y.asDenseMatrix.t)

  lazy val df: DataFrame = data(*, ::).iterator
    .map(x => (x(0), x(1), x(2), x(3)))
    .toSeq
    .toDF("x1", "x2", "x3", "label")

  lazy val assembler: VectorAssembler = new VectorAssembler()
    .setInputCols(Array("x1", "x2", "x3"))
    .setOutputCol("features")

  lazy val _df: DataFrame = assembler
    .transform(df)
    .select("features", "label")
}
