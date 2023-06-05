import numpy as np

# Algorithm hyperparameters
alpha = 0.0
epsilon = 0.1
threshold = 10

# Prisoner's Dilemma as specified by Leibo et al. Multi-agent Reinforcement Learning in Sequential Social Dilemmas https://arxiv.org/abs/1702.03037
prisoners_dilemma = np.array([
    [[3, 3], [0, 4]],
    [[4, 0], [1, 1]]
])
# Prisoner's Dilemma as specified by Apt et al. Selfishness Level of Strategic Games https://arxiv.org/abs/1105.2432 
prisoners_dilemma_apt = np.array([
    [[1, 1], [-1, 2]],
    [[2, -1], [0, 0]]
])

# Battle of the Sexes as specified by Apt et al.
battle_of_the_sexes = np.array([
    [[2, 1], [0, 0]],
    [[0, 0], [1, 2]]
])

matching_pennies = np.array([
    [[1, -1], [-1, 1]],
    [[-1, 1], [1, -1]]
])

# --------games used for testing purposes--------
multiple_so = np.array([
    [[3, 3], [0, 4]],
    [[4, 0], [3, 3]]
])

non_square_col = np.array([
    [[3, 3], [0, 4], [1, 1]],
    [[4, 0], [1, 1], [3, 3]]
])

non_square_row = np.array([
    [[3, 3], [0, 4]],
    [[4, 0], [1, 1]],
    [[1, 1], [3, 3]]
])

non_square_col_zs = np.array([
    [[1, -1], [-1, 1], [1, -1]],
    [[-1, 1], [1, -1], [-1, 1]]
])

non_square_row_zs = np.array([
    [[1, -1], [-1, 1]],
    [[-1, 1], [1, -1]],
    [[1, -1], [-1, 1]]
])
# -----------------------------------------------


# Naive Algorithm to find the selfishness level of a strategic game (based off of Algorithm listed in overleaf doc)
def find_selfishness_level(initial_alpha, epsilon, threshold, game):
    
    # Converts strategic game to altrustic game (p_i(s) = p_i(s) + alpha * SW(s))
    def strat_to_alt(alpha, game):
        # get alpha * SW(s)
        update = alpha * sum(game.T)
        # add update to strategic game's payoffs (creating the alt-game) and return alt-game
        return (game.T + update).T

    alt_game = strat_to_alt(initial_alpha, game)
    
    # Finds social optima in the strategic game
    def find_so_locs(game):
        social_welfares = sum(game.T).flatten()        
        social_optima = np.argwhere(social_welfares == np.amax(social_welfares))
        
        # check if game is n-sum (|SO| == |Payoffs|)
        if len(social_optima) == len(social_welfares):
            # if game is n-sum, throw a warning as n-sum games have inf selfishness level
            print(f"WARNING: the current game being considered is likely an n-sum game!")
            print(f"Continue anyway? (y/n)")
            x = input()
            if x == "y" or x == "yes":
                return social_optima
            else:
                exit()
        else:
            return social_optima
    
    social_optima = find_so_locs(game)
    
    alpha = initial_alpha
    
    
    # selfishness level search loop
    while alpha < threshold:
        is_nash_row = True
        is_nash_col = True
        # ------------find out what row and column the social optima are on------------
        print(f" ---Considering: alpha = {alpha} ---")
        print(f"Altrustic game with alpha = {alpha}:")
        print(alt_game)
        dims = alt_game.shape # (rows, columns, payoffs)
        locs = [(int(i/dims[1]), i%dims[1]) for i in social_optima.flatten()]
        print(f"Number of social optima detected: {len(locs)}")
        for i in enumerate(locs):
            print(f"Social optima {i[0]}: {alt_game[i[1]]}")
        # -----------------------------------------------------------------------------

        # --------check if any of the social optima are also a nash equilibrium--------
        for so_location in locs:
            # for the column player, check each other payoff in that row to see if there are any greater that the one being considered
            for column in range(dims[0]):
                if alt_game[so_location][1] < alt_game[so_location[0]][column][1]: # this is also checking the payoffs being considered - will always be true if checking >=/<=
                    is_nash_col = False
            # for the row player, check each other payoff in that column to see if there are any greater that the one being considered
            for row in range(dims[1]):
                if alt_game[so_location][0] < alt_game[row][so_location[1]][0]:
                    is_nash_row = False
                    break
        # -----------------------------------------------------------------------------
        # if one of the social optima is anash equilibrium, break out of loop
        if is_nash_col == True and is_nash_row == True:
            break

        # increment alpha and update alt game
        alpha = np.around(alpha + epsilon, 2)
        alt_game = strat_to_alt(alpha, game)


    # print out selfishness level
    print(f"-------------------- Selfishness level is {alpha} --------------------")

if __name__ == '__main__':
    find_selfishness_level(alpha, epsilon, threshold, prisoners_dilemma_apt) # <--- input the game to test here
