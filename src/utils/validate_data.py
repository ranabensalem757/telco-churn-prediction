import great_expectations as ge
from typing import Tuple , List

def validate_telco_data(df) -> Tuple[bool , List[str]]:
    ge_df = ge.from_pandas(df)
    validation_results = ge_df.validate()
    success = validation_results.success
    failed_expectations = [result['expectation_config']['expectation_type'] for result in validation_results.results if not result['success']]
    return success , failed_expectations