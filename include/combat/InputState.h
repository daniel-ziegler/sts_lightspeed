//
// Created by gamerpuppy on 7/4/2021.
//

#ifndef STS_LIGHTSPEED_INPUTSTATE_H
#define STS_LIGHTSPEED_INPUTSTATE_H

namespace sts {

    enum class InputState {
        EXECUTING_ACTIONS,

        // player choice actions
        PLAYER_NORMAL,
        CARD_SELECT,

        CHOOSE_STANCE_ACTION, // from stance potion
        CHOOSE_DISCARD_CARDS,
        SCRY,

        // random actions
        SHUFFLE_DISCARD_TO_DRAW,

        CREATE_RANDOM_CARD_IN_HAND_POWER,
        CREATE_RANDOM_CARD_IN_HAND_COLORLESS,

        SELECT_ENEMY_THE_SPECIMEN_APPLY_POISON,
    };

}


#endif //STS_LIGHTSPEED_INPUTSTATE_H
