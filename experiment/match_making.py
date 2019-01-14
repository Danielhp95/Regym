import os
import signal
import logging

from collections import namedtuple
from collections import Counter
from queue import Empty
from torch.multiprocessing import Process

from benchmark_match_play import benchmark_match_play_process

RecordedPolicy = namedtuple('RecordedPolicy', 'iteration training_scheme policy')
BenchmarkingJob = namedtuple('BenchmarkingJob', 'iteration recorded_policy_vector')


def match_making_process(expected_number_of_policies, benchmarking_episodes, createNewEnvironment, policy_queue, matrix_queue, pool):
    """
    Process in charge of keeping a record of the trained policies
    received from the training processes through the policy queue.
    This process creates a benchmarking processes for each combination
    of RecordedPolicies (iteration, training scheme, policy)
    It shuts itself down once it doesn't expect any new recorded policies.

    :param expected_number_of_policies: Number of policies that the process will wait for before shuting itself down
    :param benchmarking_episodes: Number of episodes that each benchmarking process will run for
    :param createNewEnvironment: OpenAI gym environment creation function
    :param policy_queue: Queue reference shared among processes to submit policies that will be benchmarked
    :param matrix_queue: Queue reference sent to benchmarking process, where it will put the bencharmking result
    :param pool: ProcessPoolExecutor shared between benchmarking_jobs to carry out benchmarking matches
    """
    logger = logging.getLogger('MatchMaking')
    logger.setLevel(logging.DEBUG)
    logger.info('Started')

    # Initialize variables
    received_policies = 0
    recorded_policies = []
    recorded_benchmarking_jobs = []

    benchmarking_child_processes = []
    while True:
        logger.info("Polling for policy {}: ...".format(received_policies))
        iteration, training_scheme, policy = policy_queue.get() # wait_for_policy(policy_queue)
        logger.info("Polling for policy {}: OK.".format(received_policies))
        
        received_policies += 1
        logger.info('Received ({},{},{}). {}/{} received'.format(iteration, training_scheme.name, policy.name, received_policies, expected_number_of_policies))

        recorded_policies.append(RecordedPolicy(iteration, training_scheme, policy))

        processes = [create_benchmark_process(benchmarking_episodes, createNewEnvironment,
                                              match, pool, matrix_queue, match_name)
                     for match_name, match in calculate_new_benchmarking_jobs(recorded_policies, recorded_benchmarking_jobs, iteration)]
        
        for index, p in enumerate(processes):
            logger.info("Benchmark process: {}/{} :: {}".format(index+1,len(processes),p.name))
            benchmarking_child_processes.append(p)

        logger.info("Check for termination: ...")
        check_for_termination(received_policies, expected_number_of_policies, benchmarking_child_processes, pool)
        logger.info("Check for termination: OK.")
        

def check_for_termination(received_policies, expected_number_of_policies, child_processes, pool):
    """
    Checks if process should be killed because all processing has been submitted.
    That is, all expected policies have been received and all benchmarking
    child processes have been created.

    :param received_policies: Number of policies received so far
    :param expected_number_of_policies: Number of policies that the process will wait for before shuting itself down
    :param child_processes: Benchmarking jobs still running
    :param pool: ProcessPoolExecutor shared between benchmarking_jobs to carry out benchmarking matches
    """
    if received_policies >= expected_number_of_policies:
        [p.join() for p in child_processes]
        pool.shutdown()
        os.kill(os.getpid(), signal.SIGTERM)


def calculate_new_benchmarking_jobs(recorded_policies, recorded_benchmarking_jobs, iteration_filter):
    """
    Given the current set of recorded policies,
    checks which ones haven't been been benchmarked against one another
    and computes new jobs to be sent for execution.

    :param recorded_policies: array of received RecordedPolicy
    :param recorded_benchmarking_jobs: Array of benchmarking_jobs that have kill
    :param iteration filter: Filters recorded policies on iteration because policies can only be benchmarked against those of same iteration
    """
    filtered_recorded_policies = [recorded_policy for recorded_policy in recorded_policies if recorded_policy.iteration == iteration_filter]
    for recorded_policy_1 in filtered_recorded_policies:
        for recorded_policy_2 in filtered_recorded_policies:
            benchmark_name = 'Benchmark:({},{}) vs ({},{}). iteration: {}'.format(recorded_policy_1.training_scheme.name,
                                                                                  recorded_policy_1.policy.name,
                                                                                  recorded_policy_2.training_scheme.name,
                                                                                  recorded_policy_2.policy.name,
                                                                                  iteration_filter)
            job = BenchmarkingJob(iteration_filter, [recorded_policy_1, recorded_policy_2])
            if not is_benchmarking_job_already_recorded(job, recorded_benchmarking_jobs):
                recorded_benchmarking_jobs.append(job)
                yield benchmark_name, job


def is_benchmarking_job_already_recorded(job, recorded_benchmarking_jobs):
    """
    Checks if BenchmarkingJob is a contained within the list of already
    recorded BenchmarkingJobs. For this an equality onf BenchmarkingJobs is defined
    :param job: BenchmarkingJob which is being checked to see if it has already been recorded
    :param recorded_benchmarking_jobs: List of recoreded BenchmarkingJobs
    """
    job_equality = lambda job1, job2: job1.iteration == job2.iteration and Counter(job1.recorded_policy_vector) == Counter(job2.recorded_policy_vector)
    return any([job_equality(job, recorded_job) for recorded_job in recorded_benchmarking_jobs])


def create_benchmark_process(benchmarking_episodes, createNewEnvironment, benchmark_job, pool, matrix_queue, name):
    """
    Creates a benchmarking process for the precomputed benchmark_job.
    The results of the benchmark will be put in the matrix_queue to populate confusion matrix

    :param benchmarking_episodes: Number of episodes that each benchmarking process will run for
    :param createNewEnvironment: OpenAI gym environment creation function
    :param benchmark_job: precomputed BenchmarkingJob
    :param pool: ProcessPoolExecutor shared between benchmarking_jobs to carry out benchmarking matches
    :param matrix_queue: Queue reference sent to benchmarking process, where it will put the bencharmking result
    :param name: BenchmarkingJob name identifier
    """
    benchmark_process = Process(target=benchmark_match_play_process,
                                args=(benchmarking_episodes, createNewEnvironment,
                                      benchmark_job, pool, matrix_queue, name))
    benchmark_process.start()
    return benchmark_process
