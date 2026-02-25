from unittest.mock import call, patch

import run_pipeline


def test_scripts_order():
    user_prompt = "test prompt"

    with patch("run_pipeline.subprocess.run") as run_test:
        output = run_pipeline.run_pipeline(user_prompt)

    exec = run_pipeline.sys.executable
    assert run_test.mock_calls == [
        call([exec, "prompt_parsing.py", user_prompt], check=True),
        call([exec, "simulation.py"], check=True),
        call([exec, "depth_maps.py"], check=True),
        call([exec, "optical_flow.py"], check=True),
        call([exec, "gen_animateDiff.py"], check=True),
    ]
    assert output == "output.gif"

