import pandas as pd
from src.logger import logger
import great_expectations as gx


class DataValidation:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _build_suite(self, suite_name: str):
        """builds expectation suite"""
        context = gx.get_context()

        # data source
        data_source = context.data_sources.add_pandas(name="pandas_datasource")
        data_asset = data_source.add_dataframe_asset(name="student_data")
        batch_definition = data_asset.add_batch_definition_whole_dataframe("batch")

        # expectation suite
        suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

        # 1. column existence
        for col in ["gender", "race_ethnicity", "parental_level_of_education",
                    "lunch", "test_preparation_course", "reading_score", "writing_score"]:
            suite.add_expectation(
                gx.expectations.ExpectColumnToExist(column=col)
            )

        # 2. no nulls
        for col in ["gender", "race_ethnicity", "parental_level_of_education",
                    "lunch", "test_preparation_course", "reading_score", "writing_score"]:
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
            )

        # 3. value ranges
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="reading_score", min_value=0, max_value=100
            )
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="writing_score", min_value=0, max_value=100
            )
        )

        # 4. allowed categories
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="gender", value_set=["male", "female"]
            )
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="lunch", value_set=["standard", "free/reduced"]
            )
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="test_preparation_course", value_set=["none", "completed"]
            )
        )

        # validation definition
        validation_definition = context.validation_definitions.add(
            gx.ValidationDefinition(
                name=f"{suite_name}_validation",
                data=batch_definition,
                suite=suite,
            )
        )

        return validation_definition

    def _run(self, suite_name: str) -> bool:
        validation_definition = self._build_suite(suite_name)
        results = validation_definition.run(
            batch_parameters={"dataframe": self.df}
        )

        if results["success"]:
            return True
        else:
            failed = [
                r["expectation_config"]["type"]
                for r in results["results"]
                if not r["success"]
            ]
            logger.warning(f"Validation failed for: {failed}")
            return False

    def run_validation(self) -> bool:
        """training phase"""
        try:
            logger.info("Starting training data validation")
            result = self._run("training_suite")
            if result:
                logger.info("Training data validation passed")
            return result
        except Exception as e:
            logger.exception(f"Error during training validation: {e}")
            raise

    def run_inference_validation(self) -> bool:
        """inference phase"""
        try:
            logger.info("Starting inference data validation")
            result = self._run("inference_suite")
            if result:
                logger.info("Inference data validation passed")
            return result
        except Exception as e:
            logger.exception(f"Error during inference validation: {e}")
            raise