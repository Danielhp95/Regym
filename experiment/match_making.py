import os
import signal
import logging

from collections import namedtuple
from collections import Counter
from queue import Empty
from multiprocessing import Process

from benchmark_match_play import benchmark_match_play_process

RecordedPolicy = namedtuple('RecordedPolicy', 'iteration training_scheme policy')
BenchmarkingJob = namedtuple('BenchmarkingJob', 'iteration recorded_policy_vector')


# TODO createNewEnvironment is only passed, should I have it elsewhere?
def match_making_process(expected_number_of_policies, benchmarking_episodes, createNewEnvironment, policy_queue, matrix_queue, pool):
    logger = logging.getLogger('MatchMaking')
    logger.setLevel(logging.DEBUG)
    logger.info('Started')

    # Initialize variables
    received_policies = 0
    recorded_policies = []
    benchmarking_jobs = []

    benchmarking_child_processes = []
    while True:
        iteration, training_scheme, policy = wait_for_policy(policy_queue)

        received_policies += 1
        logger.info('Received ({},{},{}). {}/{} received'.format(iteration, training_scheme.name, policy.name, received_policies, expected_number_of_policies))

        recorded_policies.append(RecordedPolicy(iteration, training_scheme, policy))

        processes = [create_bench_mark_process(benchmarking_episodes, createNewEnvironment,
                                               match, pool, matrix_queue, match_name)
                     for match_name, match in calculate_new_benchmarking_jobs(recorded_policies, benchmarking_jobs, iteration)]

        # Used to kill matchmaking process. Potential TODO to become prettier
        for p in processes:
            benchmarking_child_processes.append(p)

        check_for_termination(received_policies, expected_number_of_policies, benchmarking_child_processes, pool)


def check_for_termination(received_policies, expected_number_of_policies, child_processes, pool):
    if received_policies == expected_number_of_policies:
        pool.shutdown()
        [p.join() for p in child_processes]
        os.kill(os.getpid(), signal.SIGTERM)


def wait_for_policy(policy_queue):
    while True:
        try:
            return policy_queue.get(timeout=1)
        except Empty:
            pass


def calculate_new_benchmarking_jobs(recorded_policies, benchmarking_jobs, iteration_filter):
    for recorded_policy_1 in recorded_policies:
        for recorded_policy_2 in recorded_policies:
            benchmark_name = 'Benchmark:({},{}) vs ({},{}). iteration: {}'.format(recorded_policy_1.training_scheme.name,
                                                                                  recorded_policy_1.policy.name,
                                                                                  recorded_policy_2.training_scheme.name,
                                                                                  recorded_policy_2.policy.name,
                                                                                  iteration_filter)
            job = BenchmarkingJob(iteration_filter, [recorded_policy_1, recorded_policy_2])
            if not is_benchmarking_job_already_recorded(job, benchmarking_jobs):
                benchmarking_jobs.append(job)
                yield benchmark_name, job


# TODO Code monkey very diligent, his output not elegant
def is_benchmarking_job_already_recorded(job, benchmarking_jobs):
    job_equality = lambda job1, job2: job1.iteration == job2.iteration and Counter(job1.recorded_policy_vector) == Counter(job2.recorded_policy_vector)
    return any([job_equality(job, recorded_job) for recorded_job in benchmarking_jobs])


def create_bench_mark_process(benchmarking_episodes, createNewEnvironment, benchmark_job, pool, matrix_queue, name):
    benchmark_process = Process(target=benchmark_match_play_process,
                                args=(benchmarking_episodes, createNewEnvironment,
                                      benchmark_job, pool, matrix_queue, name))
    benchmark_process.start()
    return benchmark_process
