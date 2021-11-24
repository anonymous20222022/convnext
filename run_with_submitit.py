import argparse
import os
import uuid
from pathlib import Path

import run_classification as classification
import submitit


def parse_args():
    classification_parser = classification.get_args(submitit=True)
    parser = argparse.ArgumentParser("Submitit", parents=[classification_parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=36, type=int, help="Duration of the job, in hours")
    parser.add_argument("--job_dir", default="", type=str, help="Job folder.")
    parser.add_argument("--job_name", default="submitit", type=str, help="Job name. Must specify.")

    parser.add_argument("--partition", default="devlab", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', default=True, help="Big models? Use this")

    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()


def get_shared_folder() -> Path:
    # user = os.getenv("USER")
    if Path("checkpoint/").is_dir():
        # os.path.abspath("checkpoint/convnext")
        p = Path(os.path.abspath("checkpoint/convnext"))
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import run_classification as classification

        if self.args.enable_deepspeed:
            try:
                import deepspeed
                from deepspeed import DeepSpeedConfig
                ds_init = deepspeed.initialize
                print("Running with submitit, no additional DeepSeeed arguments will be parsed")
            except:
                print("Please 'pip install deepspeed==0.4.0'")
                exit(0)
        else:
            ds_init = None
        self._setup_gpu_args()
        classification.main(self.args, ds_init)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        # checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        # if os.path.exists(checkpoint_file):
            # self.args.resume = checkpoint_file
        self.args.auto_resume = True
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        # if "%j" in self.args.output_dir:
            # self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.output_dir = Path(self.args.job_dir)
        print(self.args.output_dir)
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    # if args.job_dir == "":
        # args.job_dir = get_shared_folder() / "%j"
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / args.job_name 
    else:
        args.job_dir = get_shared_folder() / args.job_dir

    # Note that the folder will depend on the job name, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout * 60

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name=args.job_name)

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
