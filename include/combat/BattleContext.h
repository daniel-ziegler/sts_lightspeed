//
// Created by gamerpuppy on 7/4/2021.
//

#ifndef STS_LIGHTSPEED_BATTLECONTEXT_H
#define STS_LIGHTSPEED_BATTLECONTEXT_H

#include <vector>
#include <array>

#include "sts_common.h"

#include "data_structure/fixed_list.h"
#include "constants/Potions.h"
#include "constants/MonsterEncounters.h"
#include "combat/InputState.h"
#include "combat/Player.h"
#include "combat/Monster.h"
#include "combat/MonsterGroup.h"
#include "combat/ActionQueue.h"
#include "combat/CardQueue.h"
#include "combat/Actions.h"
#include "combat/CardManager.h"
#include "combat/CardSelectInfo.h"


namespace sts {



    enum class Outcome {
        UNDECIDED=0,
        PLAYER_VICTORY,
        PLAYER_LOSS,
    };

    static constexpr const char * battleOutcomeStrings[] {
            "UNDECIDED",
            "PLAYER_VICTORY",
            "PLAYER_LOSS",
    };

    class GameContext;

    extern thread_local BattleContext *g_debug_bc;

    struct BattleContext {

        // begin for debugging purposes
        inline static int sum = 0; // for preventing optimization in benchmarks
        bool haveUsedDiscoveryAction = false; // for tracking undefined behavior resulting from using the action
        bool undefinedBehaviorEvoked = false; // some cards cause inconsistent outcomes in games
        std::uint64_t seed = 0;
        int floorNum = 0;
        MonsterEncounter encounter = MonsterEncounter::INVALID;
        int loopCount = 0;
        int movesThisTurn = 0;
        int energyWasted = 0;
        int cardsDrawn = 0;
        // Counts EmptyDeckShuffle resolutions (discard -> draw pile reshuffles). The live bridge
        // reads it around a driven advance: an outcome that rests on a reshuffle the engine rolled
        // with its own rng (e.g. an empty-pile Havoc) is unverifiable rather than a mis-simulation.
        int emptyDeckShuffleCount = 0;
        // end for debugging purposes

        Random rng;

        // Runic Dome: monster intents are hidden from the player, so move rolls defer their rng
        // until the move becomes observable (Monster::rollMove / materializePendingMove). Set
        // from the GameContext's relics before monsters init (player.relicBits is populated too
        // late for the first-turn rolls). Constant within a battle, so state hashing/equality
        // don't need it.
        bool intentsHidden = false;

        int ascension = 0;
        Outcome outcome = Outcome::UNDECIDED;
        InputState inputState = InputState::EXECUTING_ACTIONS;
        CardSelectInfo cardSelectInfo;

        int monsterTurnIdx = 6;

        bool isBattleOver = false;
        bool endTurnQueued = false;
        bool turnHasEnded = false;
        bool skipMonsterTurn = false;
        bool smokeBombUsed = false;

        ActionQueue<64> actionQueue;   // headroom for actions that push N items rather than batching
        CardQueue cardQueue;

        int potionCount = 0;
        int potionCapacity = 3;
        std::array<Potion, MAX_POTION_CAPACITY> potions;

        int turn = 0;
        Player player;
        MonsterGroup monsters;
        CardManager cards;

        CardQueueItem curCardQueueItem;

        std::bitset<32> miscBits; // 0 stolen gold check,

        const GameContext *gameContext = nullptr;

        BattleContext() = default;
        BattleContext(const BattleContext &rhs) = default;
        BattleContext(BattleContext &&rhs) = default;
        BattleContext &operator=(const BattleContext &rhs) = default;
        BattleContext &operator=(BattleContext &&rhs) = default;

// ****************************************

        void init_empty(const GameContext &gc);
        void init(const GameContext &gc);
        void init(const GameContext &gc, MonsterEncounter encounterToInit);

        void initRelics(const GameContext &gc);

        // Register the player's relics (the relicBits ownership mask only) from the GameContext,
        // WITHOUT firing any atBattleStart effects. For reconstructing a mid-combat state (the
        // CommunicationMod bridge): combat-start relic effects already happened in the real game
        // and arrive as transmitted player powers, but the ownership bits are needed so
        // during-combat relic triggers (Kunai, Shuriken, Pen Nib, Ornamental Fan, ...) fire.
        void registerRelicsFrom(const GameContext &gc);

        void exitBattle(GameContext &g) const;
        void updateRelicsOnExit(GameContext &g) const;
        void updateCardsOnExit(Deck &d) const; // for cards like ritual dagger, and eventually lesson learned results

        // HP the player will hold after exiting this battle victorious. Winning the act-boss ROOM
        // fight triggers the act-transition heal (see GameContext::transitionToAct): full heal below
        // ascension 5, 75% of missing HP at 5+, no heal with Mark of the Bloom. Every other battle
        // leaves HP unchanged -- including a boss ENCOUNTER fought in an event room (Mind Bloom's
        // I Am War), which ends with rewards and no heal.
        [[nodiscard]] int postBattleHealedHp() const;

// ****************************************

        void setRequiresStolenGoldCheck(bool value);
        [[nodiscard]] bool requiresStolenGoldCheck() const;
        [[nodiscard]] int getMonsterTurnNumber() const;  // returns 1 for first turn, 2 for second, etc.

// ****************************************

        void executeActions();

        void playCardQueueItem(CardQueueItem);
        void useCard();
        void useNoTriggerCard();

        void useAttackCard();
        void useSkillCard();
        void usePowerCard();

        void onUseAttackCard();
        void onUseSkillCard();
        void onUsePowerCard();

        void onUseStatusOrCurseCard();
        void onAfterUseCard();

        void setState(InputState state);
        void addToTop(const Action &a);
        void addToBot(const Action &a);

        void addToTopCard(CardQueueItem item);
        void addToBotCard(CardQueueItem item);

        void checkCombat();
        void clearPostCombatActions();
        void cleanCardQueue();

        [[nodiscard]] bool isCardPlayAllowed() const;

        // **********************

        void endTurn(); // called when player clicks the end turn button
        void callEndOfTurnActions(); // GameActionManager.callEndTurnActions(), called when the end turn cardQueue item is reached.
        void onTurnEnding(); // AbstractRoom endTurn(), called when the turn actually ends.

        void callEndTurnEarlySequence(); // time eater

        void applyEndOfRoundPowers(); // game : MonsterGroup applyEndOfTurnPowers
        void afterMonsterTurns();

        void obtainPotion(Potion p);
        void discardPotion(int idx);
        void drinkPotion(int idx, int target=0);
        // Smoke Bomb can't flee: Java SmokeBomb.canUse() blocks it if any monster.type == BOSS or
        // any monster has BackAttack (act-4 SURROUNDED, which drops once either of Shield/Spear
        // dies). The escape itself fires in ANY room whose phase is COMBAT -- event-spawned fights
        // (Colosseum etc.) included -- so those two checks are the whole rule. Gates drinking,
        // search enumeration, and rollout policies alike.
        [[nodiscard]] bool smokeBombEscapeBlocked() const;

        void drawCards(int count);
        void discardAtEndOfTurn();
        void discardAtEndOfTurnHelper();

        void playTopCardInDrawPile(int monsterTargetIdx, bool exhausts);

        void moveToHandHelper(CardInstance c);
        void exhaustSpecificCardInHand(int idx, std::int16_t uniqueId);
        void restoreRetainedCards(int count);
        void exhaustTopCardInHand();

        // this is
        void triggerOnEndOfTurnForPlayingCards();
        void triggerOnOtherCardPlayed(const CardInstance &usedCard);

        template <MonsterStatus s>
        void debuffEnemy(int idx, int amount, bool isSourceMonster=true);
        void debuffEnemy(MonsterStatus s, int idx, int amount, bool isSourceMonster=true);

        [[nodiscard]] int calculateCardDamage(const CardInstance &card, int targetIdx, int baseDamage) const;
        // Player-side half of the damage formula (relics AtDamageModify, powers AtDamageGive incl.
        // Heavy Blade's extra strength multiples, stance): shared by calculateCardDamage (which then
        // applies target-side modifiers) and getCardDamageDisplay (no target).
        [[nodiscard]] float applyPlayerDamageModifiers(const CardInstance &card, float damage) const;
        // Base attack damage for a card (before strength/vulnerable/etc.), accounting for the
        // combat-state-dependent cards whose base isn't the printed constant (Perfect Strike's
        // strikeCount bonus, Body Slam's block). Returns -1 for non-attack cards. Mirrors the live
        // game's AbstractCard.baseDamage so a reconstruction can be checked against the displayed card.
        [[nodiscard]] int getCardBaseDamage(const CardInstance &card) const;
        // The card's damage as the live game displays it in hand: base through the player-side
        // modifiers (strength/vigor/weak/stance/relics), NO target (no vulnerable). Mirrors live
        // AbstractCard.damage; -1 for non-attacks. Used to verify a reconstruction.
        [[nodiscard]] int getCardDamageDisplay(const CardInstance &card) const;
        [[nodiscard]] int calculateCardBlock(int baseBlock) const;

        void queuePurgeCard(const CardInstance &c, int target);
        void addPurgeCardToCardQueue(const CardQueueItem &item); // not really the front but hey
        void noOpRollMove(); // called by monsters to manipulate the rng counter when their rollMove function doesn't do anything

        void onManualDiscard(const CardInstance &c);
        void onShuffle();
        void triggerAndMoveToExhaustPile(CardInstance c);
        void mummifiedHandOnUsePower();

        // card select screens
        void openDiscoveryScreen(std::array<CardId, 3> discoveryCards, int copyCount, bool setCostToZero = true);
        void openSimpleCardSelectScreen(CardSelectTask task, int count);

        // single card select helpers
        void chooseArmamentsCard(int handIdx);
        void chooseCodexCard(CardId id);
        void chooseDiscardToHandCard(int discardIdx, bool forZeroCost);
        void chooseDiscoveryCard(CardId id);
        void chooseDualWieldCard(int handIdx);
        void chooseExhaustOneCard(int handIdx);
        void chooseExhumeCard(int exhaustIdx);
        void chooseForethoughtCard(int handIdx);
        void chooseHeadbuttCard(int discardIdx);
        void chooseRecycleCard(int handIdx);
        void chooseWarcryCard(int handIdx);

        // multi card helpers
        void chooseDrawToHandCards(const int *idxs, int cardCount);
        void chooseExhaustCards(const fixed_list<int,10> &idxs);
        void chooseGambleCards(const fixed_list<int,10> &idxs);

        bool operator==(const BattleContext &rhs) const;

        // Equality for MCTS transposition. Compares gameplay state but ignores the
        // RNG stream and pure-debug counters: two states that differ only in their
        // RNG have the same distribution of futures, so the search must treat them
        // as the same node (randomness is averaged over at chance nodes).
        [[nodiscard]] bool equalForSearch(const BattleContext &rhs) const;
    };

    std::ostream& operator<<(std::ostream &os, const BattleContext &bc);
}

#endif //STS_LIGHTSPEED_BATTLECONTEXT_H
