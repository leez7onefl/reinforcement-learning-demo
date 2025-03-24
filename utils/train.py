import os
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

log_dir = "./logs/"

def train_model(model_class, env, model_name, total_timesteps, **kwargs):
    model_log_dir = os.path.join(log_dir, model_name)
    os.makedirs(model_log_dir, exist_ok=True)

    # Set default values for keyword arguments
    kwargs.setdefault('tensorboard_log', model_log_dir)
    kwargs.setdefault('policy', 'MlpPolicy')

    # Initialize the model with the given parameters
    model = model_class(env=env, verbose=1, **kwargs)

    # Define an evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(model_log_dir, "best_model"),
        log_path=os.path.join(model_log_dir, "eval"),
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    # Log additional information to TensorBoard
    class TensorboardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            # Example: log the current time step to TensorBoard
            self.logger.record('time/total_timesteps', self.num_timesteps)
            return True

    # Learn with both EvalCallback and the custom TensorboardCallback
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, TensorboardCallback()])

    # Save the final model
    model.save(os.path.join(model_log_dir, f"{model_name}_final"))

    return model