from __future__ import print_function
import neat
import visualize  # download from the neat github
import pickle
import multiprocessing

from model import evaluate
from track import Track

# evaluate all genomes of a population
def evaluate_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        model_output = evaluate([steering_net, net], track, pedal=pedal, a_max=a_max, dt=dt, plot=False, laps=laps,
                                plot_sensor=plot_sensor, start_pos=start_pos, distance_step=distance_step,
                                sensor_settings=sensor_settings)
        genome.fitness = model_output

# evaluate single genome, used for parallel processing
def evaluate_parallel(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    model_output = evaluate([steering_net, net], track, pedal=pedal, a_max=a_max, dt=dt, plot=False, laps=laps,
                            plot_sensor=plot_sensor, start_pos=start_pos, distance_step=distance_step,
                            sensor_settings=sensor_settings)
    return model_output

# run evolutionary algorithm
def evolve(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    if start_from_checkpoint is None:
        p = neat.Population(config)
    else:
        p = neat.Checkpointer.restore_checkpoint("checkpoints/c"+str(start_from_checkpoint))

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix="checkpoints/c"))


    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), evaluate_parallel)
    winner = p.run(pe.evaluate, n_generations)  # run parallel
    #winner = p.run(evaluate_genomes, n_generations)  # run single

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    # save winner net
    with open('winner_net', 'wb') as handle:
        pickle.dump(winner_net, handle, protocol=pickle.HIGHEST_PROTOCOL)

    evaluate([steering_net, winner_net], track, pedal=pedal, a_max=a_max, dt=dt, plot=True, laps=laps,
             plot_sensor=plot_sensor, start_pos=start_pos, distance_step=distance_step, sensor_settings=sensor_settings)

    node_names = {-1: "SLL", -2: "SL", -3: "SC", -4: "SR", -5: "SRR", -6: "V", 0: "Steering", 1: "Pedal"}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    track = Track(mode=3)

    # model settings
    a_max = 10
    dt = 0.3
    distance_step = None
    laps = True
    plot_sensor = False
    start_pos = -65, 0
    pedal = True  # control steering only

    # angles, view distances. Change number of inputs in config file if number of sensors is changed.
    sensor_settings = [[-0.6, 0, 0.6], [15, 100, 15]]

    # Fitness function can be changed towards the end of model.py

    # algorithm settings
    n_generations = 500
    start_from_checkpoint = None

    double_net = False  # use separate network for steering and pedal

    # load an existing steering network for evolving a separate pedal network
    if double_net:
        with open('steering_net_dt', 'rb') as handle:
            steering_net = pickle.load(handle)
    else:
        steering_net = None

    evolve("config2")