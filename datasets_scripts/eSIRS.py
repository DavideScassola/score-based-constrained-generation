import gillespy2
import numpy as np
import tqdm


def sample_random_partition(*, total: int, partitions: int) -> np.ndarray:
    x = np.random.dirichlet(alpha=np.ones(partitions), size=1).squeeze()
    x = np.round(x * total)
    x[-1] = total - np.sum(x[:-1])
    return x


class eSIRS(gillespy2.Model):
    def __init__(self, total_population, endtime, n_steps):
        # First call the gillespy2.Model initializer.
        super().__init__("eSIRS")
        # Define parameters for the rates of creation and dissociation.
        k1 = gillespy2.Parameter(name="k1", expression=2.36012158 / total_population)
        k2 = gillespy2.Parameter(name="k2", expression=1.6711464)
        k3 = gillespy2.Parameter(name="k3", expression=0.90665231)
        k4 = gillespy2.Parameter(name="k4", expression=0.63583386)

        self.add_parameter([k1, k2, k3, k4])
        self.total_population = total_population
        # Define variables for the molecular species representing M & D.
        S = gillespy2.Species(name="S", initial_value=total_population)
        I = gillespy2.Species(name="I", initial_value=0)
        R = gillespy2.Species(name="R", initial_value=0)
        self.species_list = [S, I, R]
        self.add_species(self.species_list)

        # The list of reactants and products for a Reaction object are
        # each a Python dictionary in which the dictionary keys are
        # Species objects and the values are stoichiometries of the
        # species in the reaction.

        # infection
        r1 = gillespy2.Reaction(
            name="r1", rate=k1, reactants={S: 1, I: 1}, products={I: 2}
        )
        # recovery
        r2 = gillespy2.Reaction(name="r2", rate=k2, reactants={I: 1}, products={R: 1})

        # loss of immunity
        r3 = gillespy2.Reaction(name="r3", rate=k3, reactants={R: 1}, products={S: 1})

        # external infection
        r4 = gillespy2.Reaction(name="r4", rate=k4, reactants={S: 1}, products={I: 1})

        self.add_reaction([r1, r2, r3, r4])

        # Set the timespan for the simulation.
        self.timespan(np.linspace(0, endtime, n_steps))

    def set_initial_state(self) -> None:
        x = sample_random_partition(
            total=self.total_population, partitions=len(self.species_list)
        )
        for i in range(len(self.species_list)):
            self.species_list[i].initial_value = x[i]


if __name__ == "__main__":
    N_SAMPLES = 10_000
    STARTING_POINTS = 1000  # ideally it would be the same as the number of samples, but it's not efficient (probably there is a solution)

    total_population = 100
    n_steps = 30
    endtime = 3.2
    trajectories_per_sarting_point = N_SAMPLES // STARTING_POINTS

    model = eSIRS(total_population=total_population, n_steps=n_steps, endtime=endtime)

    trajectories = np.empty(
        (
            STARTING_POINTS,
            trajectories_per_sarting_point,
            n_steps,
            len(model.species_list) - 1,
        )
    )

    for i in tqdm.tqdm(range(STARTING_POINTS)):
        model.set_initial_state()
        x = model.run(number_of_trajectories=trajectories_per_sarting_point)
        for j in range(trajectories_per_sarting_point):
            trajectories[i, j, :, 0] = x[j]["S"]
            trajectories[i, j, :, 1] = x[j]["I"]

    X = trajectories.reshape(
        (STARTING_POINTS * trajectories_per_sarting_point, n_steps, -1)
    )

    np.random.shuffle(X)

    filename = f"data/eSIRS.npy"

    np.save(filename, arr=X, allow_pickle=False)
