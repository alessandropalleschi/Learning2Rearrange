"""Build the trainer of the configured agent (`Config.agent`): dense Q-maps, object-centric Q, or policy heatmap + critic."""
from l2r.learning.trainer import Trainer


def make_trainer(cfg, *args, **kwargs):
    """Trainer / ObjectTrainer / PolicyCriticTrainer, all with the same interface (`forward_batch` returns dense Q-maps)."""
    if cfg.agent == 'object':
        if cfg.object_algo == 'iql':
            from l2r.learning.iql_trainer import IQLObjectTrainer
            return IQLObjectTrainer(cfg, *args, **kwargs)
        from l2r.learning.object_trainer import ObjectTrainer
        return ObjectTrainer(cfg, *args, **kwargs)
    if cfg.agent == 'policy_critic':
        from l2r.learning.policy_critic import PolicyCriticTrainer
        return PolicyCriticTrainer(cfg, *args, **kwargs)
    return Trainer(cfg, *args, **kwargs)
