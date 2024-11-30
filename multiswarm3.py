from particle import Particle
from evaluation import evaluate

def optimize_schedule(data, max_iterations=100, num_particles=10):
    """
    Orchestrates the multi-swarm PSO optimization.
    """
    swarms = [[Particle(generate_initial_schedule(data)) for _ in range(num_particles)]]

    for iteration in range(max_iterations):
        for swarm in swarms:
            for particle in swarm:
                # Update particle
                particle.update(
                    data,
                    personal_best=particle.best,
                    global_best=min(swarm, key=lambda p: p.fitness),
                    chi=0.729, c1=1.0, c2=1.0
                )

                # Evaluate fitness
                fitness = evaluate(
                    particle.schedule, data["courses"], data["rooms"], data["curricula"], data["constraints"]
                )
                particle.fitness = fitness

                # Update personal best
                if fitness < particle.best_fitness:
                    particle.best = particle.schedule.copy()
                    particle.best_fitness = fitness

    # Return the best solution across all swarms
    best_swarm = min(swarms, key=lambda s: min(p.fitness for p in s))
    best_particle = min(best_swarm, key=lambda p: p.fitness)
    return best_particle.schedule
