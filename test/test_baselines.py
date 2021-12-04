import os
import sys
import pytest



# from unstable_baselines.baselines.dqn.main import main as dqn_main
# from unstable_baselines.baselines.ppo.main import main as ppo_main
# from unstable_baselines.baselines.sac.main import main as sac_main
# from unstable_baselines.baselines.ddpg.main import main as ddpg_main
# from unstable_baselines.baselines.redq.main import main as redq_main
# from unstable_baselines.baselines.tdn_sac.main import main as tdn_sac_main

try:    
    from unstable_baselines.baselines.dqn.main import main as dqn_main
    from unstable_baselines.baselines.ppo.main import main as ppo_main
    from unstable_baselines.baselines.sac.main import main as sac_main
    from unstable_baselines.baselines.ddpg.main import main as ddpg_main
    from unstable_baselines.baselines.redq.main import main as redq_main
    from unstable_baselines.baselines.tdn_sac.main import main as tdn_sac_main
except:
    dqn_main = None
    ppo_main = None
    sac_main = None
    ddpg_main = None
    redq_main = None
    tdn_sac_main = None


@pytest.mark.skipif(dqn_main is None, reason='')
@pytest.mark.parametrize('config_path', ['unstable_baselines/baselines/dqn/configs/test.json'])
def test_dqn(config_path):
    with pytest.raises(SystemExit) as e:
        args = [config_path]
        dqn_main(args)
    exec_msg = e.value.args[0]
    assert exec_msg == 0


@pytest.mark.skipif(ppo_main is None, reason='')
@pytest.mark.parametrize('config_path', ['unstable_baselines/baselines/ppo/configs/test.json'])
def test_ppo(config_path):
    with pytest.raises(SystemExit) as e:
        args = [config_path]
        ppo_main(args)
    exec_msg = e.value.args[0]
    assert exec_msg == 0


@pytest.mark.skipif(sac_main is None, reason='')
@pytest.mark.parametrize('config_path', ["unstable_baselines/baselines/sac/configs/test.json"])
def test_sac(config_path):
    with pytest.raises(SystemExit) as e:
        args = [config_path]
        sac_main(args)
    exec_msg = e.value.args[0]
    assert exec_msg == 0

@pytest.mark.skipif(ddpg_main is None, reason='')
@pytest.mark.parametrize('config_path', ['unstable_baselines/baselines/ddpg/configs/test.json'])
def test_ddpg(config_path):
    with pytest.raises(SystemExit) as e:
        args = [config_path]
        ddpg_main(args)
    exec_msg = e.value.args[0]
    assert exec_msg == 0


@pytest.mark.skipif(redq_main is None, reason='')
@pytest.mark.parametrize('config_path', ['unstable_baselines/baselines/redq/configs/test.json'])
def test_redq(config_path):
    with pytest.raises(SystemExit) as e:
        args = [config_path]
        redq_main(args)
    exec_msg = e.value.args[0]
    assert exec_msg == 0


@pytest.mark.skipif(tdn_sac_main is None, reason='')
@pytest.mark.parametrize('config_path', ['unstable_baselines/baselines/tdn_sac/configs/test.json'])
def test_tdn_sac(config_path):
    with pytest.raises(SystemExit) as e:
        args = [config_path]
        tdn_sac_main(args)
    exec_msg = e.value.args[0]
    assert exec_msg == 0