from unstable_baselines.envs.mujoco_meta.gym.scoreboard.client.tests import helper
from unstable_baselines.envs.mujoco_meta.gym import scoreboard

class EvaluationTest(helper.APITestCase):
    def test_create_evaluation(self):
        self.mock_response(helper.TestData.evaluation_response())

        evaluation = scoreboard.Evaluation.create()
        assert isinstance(evaluation, scoreboard.Evaluation)

        self.requestor_mock.request.assert_called_with(
            'post',
            '/v1/evaluations',
            {},
            None
        )
