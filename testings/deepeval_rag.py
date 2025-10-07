from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric


def evaluate_rag_response(final_state_value, distinct_search_results):
    # Define the metric with a threshold
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini")
    hallucination_metric = HallucinationMetric(threshold=0.5)

    test_relevancy = LLMTestCase(
        input=final_state_value.values["query"],
        actual_output=final_state_value.values["final_result"].content,
        retrieval_context=[result["content"] for result in distinct_search_results],
    )

    test_hallucination = LLMTestCase(
        input=final_state_value.values["query"],
        actual_output=final_state_value.values["final_result"].content,
        context=[result["content"] for result in distinct_search_results],
    )

    assert_test(test_relevancy, [relevancy_metric])
    assert_test(test_hallucination, [hallucination_metric])
