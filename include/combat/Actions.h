//
// Created by gamerpuppy on 7/4/2021.
//

#ifndef STS_LIGHTSPEED_ACTIONS_H
#define STS_LIGHTSPEED_ACTIONS_H

#include <utility>
#include <functional>
#include <variant>
#include <iostream>
#include <cstdint>

#include "constants/PlayerStatusEffects.h"
#include "constants/MonsterStatusEffects.h"
#include "constants/Cards.h"
#include "constants/Potions.h"

#include "combat/InputState.h"
#include "combat/CardSelectInfo.h"
#include "combat/CardInstance.h"
#include "combat/CardQueue.h"

namespace sts {

    class BattleContext;
    class CardQueueItem;

    typedef std::array<std::uint16_t,5> DamageMatrix;

    struct _SetState { InputState state; void operator()(BattleContext& bc) const; };
    
    struct _BuffPlayer { PlayerStatus s; int amount = 1; void operator()(BattleContext& bc) const; };
    struct _DebuffPlayer { PlayerStatus s; int amount; bool isSourceMonster = true; void operator()(BattleContext& bc) const; };

    struct _DecrementStatus { PlayerStatus s; int amount = 1; void operator()(BattleContext& bc) const; };
    struct _RemoveStatus { PlayerStatus s; void operator()(BattleContext& bc) const; };

    struct _BuffEnemy { MonsterStatus s; int idx; int amount = 1; void operator()(BattleContext& bc) const; };
    struct _DebuffEnemy { MonsterStatus s; int idx; int amount; bool isSourceMonster = true; void operator()(BattleContext& bc) const; };
    struct _DebuffAllEnemy { MonsterStatus s; int amount; bool isSourceMonster = true; void operator()(BattleContext& bc) const; };


    // todo, action and damage actions should clear certain actions from the queue if the player kills all enemies
    struct _AttackEnemy { int idx; int damage; void operator()(BattleContext& bc) const; };

    // for cards that do one hit, like cleave
    struct _AttackAllEnemy { int baseDamage; void operator()(BattleContext& bc) const; };

    // for cards like whirlwind, where damage is calculated once
    struct _AttackAllEnemyMatrix { DamageMatrix damageMatrix; void operator()(BattleContext& bc) const; };

    struct _DamageEnemy { int idx; int damage; void operator()(BattleContext& bc) const; };

    // all usages of this in game have pureDamage set to true
    struct _DamageAllEnemy { int damage; void operator()(BattleContext& bc) const; }; // todo maybe need to add boolean isPlayerOwner

    struct _AttackPlayer { int idx; int damage; void operator()(BattleContext& bc) const; };
    struct _DamagePlayer { int damage; bool selfDamage=false; void operator()(BattleContext& bc) const; };

    struct _VampireAttack { int damage; void operator()(BattleContext& bc) const; }; // only used by shelled parasite

    struct _PlayerLoseHp { int hp; bool selfDamage=false; void operator()(BattleContext& bc) const; };
    struct _HealPlayer { int amount; void operator()(BattleContext& bc) const; };

    struct _MonsterGainBlock { int idx; int amount; void operator()(BattleContext& bc) const; };
    struct _RollMove { int monsterIdx; void operator()(BattleContext& bc) const; };
    struct _ReactiveRollMove { void operator()(BattleContext& bc) const; }; // for writhing mass

    struct _NoOpRollMove { void operator()(BattleContext& bc) const; };

    struct _ChangeStance { Stance stance; void operator()(BattleContext& bc) const; };
    struct _GainEnergy { int amount; void operator()(BattleContext& bc) const; };
    struct _GainBlock { int amount; void operator()(BattleContext& bc) const; };
    struct _DrawCards { int amount; void operator()(BattleContext& bc) const; };


    // the draw pile doesn't actually have to be empty
    // the game calls on shuffle when this is created
    struct _EmptyDeckShuffle { void operator()(BattleContext& bc) const; };

    // used by Deep Breath and Reboot, does not trigger onShuffle relics
    struct _ShuffleDrawPile { void operator()(BattleContext& bc) const; };

    struct _ShuffleTempCardIntoDrawPile { CardId id; int count=1; void operator()(BattleContext& bc) const; };

    struct _PlayTopCard { int monsterTargetIdx; bool exhausts; void operator()(BattleContext& bc) const; };
    struct _MakeTempCardInHand { CardInstance card; int amount = 1; void operator()(BattleContext& bc) const; };
    struct _MakeTempCardInDrawPile { CardInstance c; int amount; bool shuffleInto; void operator()(BattleContext& bc) const; };
    struct _MakeTempCardInDiscard { CardInstance c; int amount = 1; void operator()(BattleContext& bc) const; };
    struct _MakeTempCardsInHand { std::vector<CardInstance> cards; void operator()(BattleContext& bc) const; };

    struct _DiscardNoTriggerCard { void operator()(BattleContext& bc) const; }; // for doubt, shame, etc; discards the BattleContext.curCard


    struct _ClearCardQueue { void operator()(BattleContext& bc) const; };
    struct _DiscardAtEndOfTurn { void operator()(BattleContext& bc) const; };
    struct _DiscardAtEndOfTurnHelper { void operator()(BattleContext& bc) const; };
    struct _RestoreRetainedCards { int count; void operator()(BattleContext& bc) const; };

    struct _UnnamedEndOfTurnAction { void operator()(BattleContext& bc) const; };
    struct _MonsterStartTurnAction { void operator()(BattleContext& bc) const; };
    struct _TriggerEndOfTurnOrbsAction { void operator()(BattleContext& bc) const; };

    struct _ExhaustTopCardInHand { void operator()(BattleContext& bc) const; };
    struct _ExhaustSpecificCardInHand { int idx; std::int16_t uniqueId; void operator()(BattleContext& bc) const; };

    // Random Actions

    struct _DamageRandomEnemy { int damage; void operator()(BattleContext& bc) const; }; // juggernaut
    struct _GainBlockRandomEnemy { int sourceMonster; int amount; void operator()(BattleContext& bc) const; };

    struct _SummonGremlins { void operator()(BattleContext& bc) const; };
    struct _SpawnTorchHeads { void operator()(BattleContext& bc) const; };
    struct _SpireShieldDebuff { void operator()(BattleContext& bc) const; }; // only called if player has orb slots

    struct _ExhaustRandomCardInHand { int count; void operator()(BattleContext& bc) const; };
    struct _MadnessAction { void operator()(BattleContext& bc) const; };
    struct _RandomizeHandCost { void operator()(BattleContext& bc) const; };
    struct _UpgradeRandomCardAction { void operator()(BattleContext& bc) const; }; // Warped Tongs Relic

    struct _CodexAction { void operator()(BattleContext& bc) const; }; // Nilrys Codex onPlayerEndTurn
    struct _ExhaustMany { int limit; void operator()(BattleContext& bc) const; };
    struct _GambleAction { void operator()(BattleContext& bc) const; };
    struct _ToolboxAction { void operator()(BattleContext& bc) const; };
    struct _FiendFireAction { int targetIdx; int calculatedDamage; void operator()(BattleContext& bc) const; }; // Fiend Fire Card
    struct _SwordBoomerangAction { int baseDamage; void operator()(BattleContext& bc) const; }; // Sword Boomerang

    struct _PutRandomCardsInDrawPile { CardType type; int count; void operator()(BattleContext& bc) const; }; // used by chrysalis and metamorphosis
    struct _DiscoveryAction { CardType type; int amount; void operator()(BattleContext& bc) const; }; // attack potion; skill potion
    struct _InfernalBladeAction { void operator()(BattleContext& bc) const; };
    struct _JackOfAllTradesAction { bool upgraded; void operator()(BattleContext& bc) const; };
    struct _TransmutationAction { bool upgraded; int energy; bool useEnergy; void operator()(BattleContext& bc) const; };
    struct _ViolenceAction { int count; void operator()(BattleContext& bc) const; };

    struct _BetterDiscardPileToHandAction { int amount; CardSelectTask task; void operator()(BattleContext& bc) const; }; // used by hologram and liquid memories
    struct _ArmamentsAction { void operator()(BattleContext& bc) const; };
    struct _DualWieldAction { int copyCount; void operator()(BattleContext& bc) const; };
    struct _ExhumeAction { void operator()(BattleContext& bc) const; };
    struct _ForethoughtAction { bool upgraded; void operator()(BattleContext& bc) const; };
    struct _HeadbuttAction { void operator()(BattleContext& bc) const; };
    struct _ChooseExhaustOne { void operator()(BattleContext& bc) const; };
    struct _DrawToHandAction { CardSelectTask task; CardType cardType; void operator()(BattleContext& bc) const; };
    struct _WarcryAction { void operator()(BattleContext& bc) const; };

    // ************

    struct _TimeEaterPlayCardQueueItem { CardQueueItem item; void operator()(BattleContext& bc) const; };
    struct _UpgradeAllCardsInHand { void operator()(BattleContext& bc) const; };
    struct _OnAfterCardUsed { void operator()(BattleContext& bc) const; }; // called UseCardAction in game
    struct _EssenceOfDarkness { int darkOrbsPerSlot; void operator()(BattleContext& bc) const; }; // handle sacred bark
    struct _IncreaseOrbSlots { int count; void operator()(BattleContext& bc) const; };

    struct _SuicideAction { int monsterIdx; bool triggerRelics; void operator()(BattleContext& bc) const; };

    struct _PoisonLoseHpAction { void operator()(BattleContext& bc) const; };
    struct _RemovePlayerDebuffs { void operator()(BattleContext& bc) const; }; // Orange Pellets Relic

    struct _DualityAction { void operator()(BattleContext& bc) const; }; // Duality Relic

    struct _ApotheosisAction { void operator()(BattleContext& bc) const; };
    struct _DropkickAction { int targetIdx; void operator()(BattleContext& bc) const; };
    struct _EnlightenmentAction { bool upgraded; void operator()(BattleContext& bc) const; };
    struct _EntrenchAction { void operator()(BattleContext& bc) const; };
    struct _FeedAction { int idx; int damage; bool upgraded; void operator()(BattleContext& bc) const; };
    struct _HandOfGreedAction { int idx; int damage; bool upgraded; void operator()(BattleContext& bc) const; };
    struct _LimitBreakAction { void operator()(BattleContext& bc) const; };
    struct _ReaperAction { int baseDamage; void operator()(BattleContext& bc) const; };
    struct _RitualDaggerAction { int idx; int damage; void operator()(BattleContext& bc) const; };
    struct _SecondWindAction { int blockPerCard; void operator()(BattleContext& bc) const; };
    struct _SeverSoulExhaustAction { void operator()(BattleContext& bc) const; };
    struct _SpotWeaknessAction { int target; int strength; void operator()(BattleContext& bc) const; }; // Spot Weakness
    struct _WhirlwindAction { int baseDamage; int energy; bool useEnergy; void operator()(BattleContext& bc) const; };

    struct _AttackAllMonsterRecursive { DamageMatrix matrix; int timesRemaining; void operator()(BattleContext& bc) const; };

    typedef std::variant<
        _SetState,
        _BuffPlayer,
        _DebuffPlayer,
        _DecrementStatus,
        _RemoveStatus,
        _BuffEnemy,
        _DebuffEnemy,
        _DebuffAllEnemy,
        _AttackEnemy,
        _AttackAllEnemy,
        _AttackAllEnemyMatrix,
        _DamageEnemy,
        _DamageAllEnemy,
        _AttackPlayer,
        _DamagePlayer,
        _VampireAttack,
        _PlayerLoseHp,
        _HealPlayer,
        _MonsterGainBlock,
        _RollMove,
        _ReactiveRollMove,
        _NoOpRollMove,
        _ChangeStance,
        _GainEnergy,
        _GainBlock,
        _DrawCards,
        _EmptyDeckShuffle,
        _ShuffleDrawPile,
        _ShuffleTempCardIntoDrawPile,
        _PlayTopCard,
        _MakeTempCardInHand,
        _MakeTempCardInDrawPile,
        _MakeTempCardInDiscard,
        _MakeTempCardsInHand,
        _DiscardNoTriggerCard,
        _ClearCardQueue,
        _DiscardAtEndOfTurn,
        _DiscardAtEndOfTurnHelper,
        _RestoreRetainedCards,
        _UnnamedEndOfTurnAction,
        _MonsterStartTurnAction,
        _TriggerEndOfTurnOrbsAction,
        _ExhaustTopCardInHand,
        _ExhaustSpecificCardInHand,
        _DamageRandomEnemy,
        _GainBlockRandomEnemy,
        _SummonGremlins,
        _SpawnTorchHeads,
        _SpireShieldDebuff,
        _ExhaustRandomCardInHand,
        _MadnessAction,
        _RandomizeHandCost,
        _UpgradeRandomCardAction,
        _CodexAction,
        _ExhaustMany,
        _GambleAction,
        _ToolboxAction,
        _FiendFireAction,
        _SwordBoomerangAction,
        _PutRandomCardsInDrawPile,
        _DiscoveryAction,
        _InfernalBladeAction,
        _JackOfAllTradesAction,
        _TransmutationAction,
        _ViolenceAction,
        _BetterDiscardPileToHandAction,
        _ArmamentsAction,
        _DualWieldAction,
        _ExhumeAction,
        _ForethoughtAction,
        _HeadbuttAction,
        _ChooseExhaustOne,
        _DrawToHandAction,
        _WarcryAction,
        _TimeEaterPlayCardQueueItem,
        _UpgradeAllCardsInHand,
        _OnAfterCardUsed,
        _EssenceOfDarkness,
        _IncreaseOrbSlots,
        _SuicideAction,
        _PoisonLoseHpAction,
        _RemovePlayerDebuffs,
        _DualityAction,
        _ApotheosisAction,
        _DropkickAction,
        _EnlightenmentAction,
        _EntrenchAction,
        _FeedAction,
        _HandOfGreedAction,
        _LimitBreakAction,
        _ReaperAction,
        _RitualDaggerAction,
        _SecondWindAction,
        _SeverSoulExhaustAction,
        _SpotWeaknessAction,
        _WhirlwindAction,
        _AttackAllMonsterRecursive
    > Action;

    struct Actions {
        static Action SetState(InputState state) {
            return Action(_SetState {state});
        }

        // implemented in BattleContext.h because of template
        static Action BuffPlayer(PlayerStatus s, int amount=1) {
            return Action(_BuffPlayer {s, amount});
        }
        static Action DebuffPlayer(PlayerStatus s, int amount=1, bool isSourceMonster=true) {
            return Action(_DebuffPlayer {s, amount, isSourceMonster});
        }

        static Action DecrementStatus(PlayerStatus s, int amount=1) {
            return Action(_DecrementStatus {s, amount});
        }
        static Action RemoveStatus(PlayerStatus s) {
            return Action(_RemoveStatus {s});
        }

        static Action BuffEnemy(MonsterStatus s, int idx, int amount=1) {
            return Action(_BuffEnemy {s, idx, amount});
        }
        static Action DebuffEnemy(MonsterStatus s, int idx, int amount, bool isSourceMonster=true) {
            return Action(_DebuffEnemy {s, idx, amount, isSourceMonster});
        }
        static Action DebuffAllEnemy(MonsterStatus s, int amount, bool isSourceMonster=true) {
            return Action(_DebuffAllEnemy {s, amount, isSourceMonster});
        }


        // todo, action and damage actions should clear certain actions from the queue if the player kills all enemies
        static Action AttackEnemy(int idx, int damage) {
            return Action(_AttackEnemy {idx, damage});
        }

        // for cards that do one hit, like cleave
        static Action AttackAllEnemy(int baseDamage) {
            return Action(_AttackAllEnemy {baseDamage});
        }

        // for cards like whirlwind, where damage is calculated once
        static Action AttackAllEnemy(DamageMatrix damageMatrix) {
            return Action(_AttackAllEnemyMatrix {damageMatrix});
        }

        static Action DamageEnemy(int idx, int damage) {
            return Action(_DamageEnemy {idx, damage});
        }

        // all usages of this in game have pureDamage set to true
        static Action DamageAllEnemy(int damage) { // todo maybe need to add boolean isPlayerOwner
            return Action(_DamageAllEnemy {damage});
        }

        static Action AttackPlayer(int idx, int damage) {
            return Action(_AttackPlayer {idx, damage});
        }
        static Action DamagePlayer(int damage, bool selfDamage=false) {
            return Action(_DamagePlayer {damage, selfDamage});
        }

        // only used by shelled parasite
        static Action VampireAttack(int damage) {
            return Action(_VampireAttack {damage});
        }

        static Action PlayerLoseHp(int hp, bool selfDamage=false) {
            return Action(_PlayerLoseHp {hp, selfDamage});
        }
        static Action HealPlayer(int amount) {
            return Action(_HealPlayer {amount});
        }

        static Action MonsterGainBlock(int idx, int amount) {
            return Action(_MonsterGainBlock {idx, amount});
        }
        static Action RollMove(int monsterIdx) {
            return Action(_RollMove {monsterIdx});
        }
        
        // for writhing mass
        static Action ReactiveRollMove() {
            return Action(_ReactiveRollMove {});
        }

        static Action NoOpRollMove() {
            return Action(_NoOpRollMove {});
        }

        static Action ChangeStance(Stance stance) {
            return Action(_ChangeStance {stance});
        }
        static Action GainEnergy(int amount) {
            return Action(_GainEnergy {amount});
        }
        static Action GainBlock(int amount) {
            return Action(_GainBlock {amount});
        }
        static Action DrawCards(int amount) {
            return Action(_DrawCards {amount});
        }

        // the draw pile doesn't actually have to be empty
        // the game calls on shuffle when this is created
        static Action EmptyDeckShuffle() {
            return Action(_EmptyDeckShuffle {});
        }

        // used by Deep Breath and Reboot, does not trigger onShuffle relics
        static Action ShuffleDrawPile() {
            return Action(_ShuffleDrawPile {});
        }

        static Action ShuffleTempCardIntoDrawPile(CardId id, int count=1) {
            return Action(_ShuffleTempCardIntoDrawPile {id, count});
        }

        static Action PlayTopCard(int monsterTargetIdx, bool exhausts) {
            return Action(_PlayTopCard {monsterTargetIdx, exhausts});
        }
        static Action MakeTempCardInHand(CardId card, bool upgraded=false, int amount=1) {
            return Action(_MakeTempCardInHand {CardInstance(card, upgraded), amount});
        }
        static Action MakeTempCardInHand(CardInstance c, int amount=1) {
            return Action(_MakeTempCardInHand {c, amount});
        }
        static Action MakeTempCardInDrawPile(const CardInstance &c, int amount, bool shuffleInto) {
            return Action(_MakeTempCardInDrawPile {c, amount, shuffleInto});
        }
        static Action MakeTempCardInDiscard(const CardInstance &c, int amount=1) {
            return Action(_MakeTempCardInDiscard {c, amount});
        }
        static Action MakeTempCardsInHand(std::vector<CardInstance> cards) {
            return Action(_MakeTempCardsInHand {cards});
        }

        // for doubt, shame, etc, discards the BattleContext.curCard
        static Action DiscardNoTriggerCard() {
            return Action(_DiscardNoTriggerCard {});
        }

        static Action ClearCardQueue() {
            return Action(_ClearCardQueue {});
        }
        static Action DiscardAtEndOfTurn() {
            return Action(_DiscardAtEndOfTurn {});
        }
        static Action DiscardAtEndOfTurnHelper() {
            return Action(_DiscardAtEndOfTurnHelper {});
        }
        static Action RestoreRetainedCards(int count) {
            return Action(_RestoreRetainedCards {count});
        }

        static Action UnnamedEndOfTurnAction() {
            return Action(_UnnamedEndOfTurnAction {});
        }
        static Action MonsterStartTurnAction() {
            return Action(_MonsterStartTurnAction {});
        }
        static Action TriggerEndOfTurnOrbsAction() {
            return Action(_TriggerEndOfTurnOrbsAction {});
        }

        static Action ExhaustTopCardInHand() {
            return Action(_ExhaustTopCardInHand {});
        }
        static Action ExhaustSpecificCardInHand(int idx, std::int16_t uniqueId) {
            return Action(_ExhaustSpecificCardInHand {idx, uniqueId});
        }
        
        // Random Actions

        // juggernaut
        static Action DamageRandomEnemy(int damage) {
            return Action(_DamageRandomEnemy {damage});
        }
        static Action GainBlockRandomEnemy(int sourceMonster, int amount) {
            return Action(_GainBlockRandomEnemy {sourceMonster, amount});
        }

        static Action SummonGremlins() {
            return Action(_SummonGremlins {});
        }
        static Action SpawnTorchHeads() {
            return Action(_SpawnTorchHeads {});
        }
        // only called if player has orb slots
        static Action SpireShieldDebuff() {
            return Action(_SpireShieldDebuff {});
        }

        static Action ExhaustRandomCardInHand(int count) {
            return Action(_ExhaustRandomCardInHand {count});
        }
        static Action MadnessAction() {
            return Action(_MadnessAction {});
        }
        static Action RandomizeHandCost() {
            return Action(_RandomizeHandCost {});
        }
        // Warp Tongs Relic
        static Action UpgradeRandomCardAction() {
            return Action(_UpgradeRandomCardAction {});
        }

        // Nilrys Codex onPlayerEndTurn
        static Action CodexAction() {
            return Action(_CodexAction {});
        }
        static Action ExhaustMany(int limit) {
            return Action(_ExhaustMany {limit});
        }
        static Action GambleAction() {
            return Action(_GambleAction {});
        }
        static Action ToolboxAction() {
            return Action(_ToolboxAction {});
        }
        
        // Fiend Fire Card
        static Action FiendFireAction(int targetIdx, int calculatedDamage) {
            return Action(_FiendFireAction {targetIdx, calculatedDamage});
        }
        
        // Sword Boomerang
        static Action SwordBoomerangAction(int baseDamage) {
            return Action(_SwordBoomerangAction {baseDamage});
        }

        // used by chrysalis and metamorphosis
        static Action PutRandomCardsInDrawPile(CardType type, int count) {
            return Action(_PutRandomCardsInDrawPile {type, count});
        }
        
        // attack potion, skill potion
        static Action DiscoveryAction(CardType type, int amount) {
            return Action(_DiscoveryAction {type, amount});
        }
        static Action InfernalBladeAction() {
            return Action(_InfernalBladeAction {});
        }
        static Action JackOfAllTradesAction(bool upgraded) {
            return Action(_JackOfAllTradesAction {upgraded});
        }
        static Action TransmutationAction(bool upgraded, int energy, bool useEnergy) {
            return Action(_TransmutationAction {upgraded, energy, useEnergy});
        }
        static Action ViolenceAction(int count) {
            return Action(_ViolenceAction {count});
        }

        // used by hologram and liquid memories
        static Action BetterDiscardPileToHandAction(int amount, CardSelectTask task) {
            return Action(_BetterDiscardPileToHandAction {amount, task});
        }
        static Action ArmamentsAction() {
            return Action(_ArmamentsAction {});
        }
        static Action DualWieldAction(int copyCount) {
            return Action(_DualWieldAction {copyCount});
        }
        static Action ExhumeAction() {
            return Action(_ExhumeAction {});
        }
        static Action ForethoughtAction(bool upgraded) {
            return Action(_ForethoughtAction {upgraded});
        }
        static Action HeadbuttAction() {
            return Action(_HeadbuttAction {});
        }
        static Action ChooseExhaustOne() {
            return Action(_ChooseExhaustOne {});
        }
        static Action DrawToHandAction(CardSelectTask task, CardType cardType) {
            return Action(_DrawToHandAction {task, cardType});
        }
        static Action WarcryAction() {
            return Action(_WarcryAction {});
        }

        // ************

        static Action TimeEaterPlayCardQueueItem(const CardQueueItem &item) {
            return Action(_TimeEaterPlayCardQueueItem {item});
        }
        static Action UpgradeAllCardsInHand() {
            return Action(_UpgradeAllCardsInHand {});
        }
        static Action OnAfterCardUsed() {
            return Action(_OnAfterCardUsed {});
        }
        static Action EssenceOfDarkness(int darkOrbsPerSlot) {
            return Action(_EssenceOfDarkness {darkOrbsPerSlot});
        }
        static Action IncreaseOrbSlots(int count) {
            return Action(_IncreaseOrbSlots {count});
        }

        static Action SuicideAction(int monsterIdx, bool triggerRelics) {
            return Action(_SuicideAction {monsterIdx, triggerRelics});
        }

        static Action PoisonLoseHpAction() {
            return Action(_PoisonLoseHpAction {});
        }
        static Action RemovePlayerDebuffs() {
            return Action(_RemovePlayerDebuffs {});
        }

        static Action DualityAction() {
            return Action(_DualityAction {});
        }

        static Action ApotheosisAction() {
            return Action(_ApotheosisAction {});
        }
        static Action DropkickAction(int targetIdx) {
            return Action(_DropkickAction {targetIdx});
        }
        static Action EnlightenmentAction(bool upgraded) {
            return Action(_EnlightenmentAction {upgraded});
        }
        static Action EntrenchAction() {
            return Action(_EntrenchAction {});
        }
        static Action FeedAction(int idx, int damage, bool upgraded) {
            return Action(_FeedAction {idx, damage, upgraded});
        }
        static Action HandOfGreedAction(int idx, int damage, bool upgraded) {
            return Action(_HandOfGreedAction {idx, damage, upgraded});
        }
        static Action LimitBreakAction() {
            return Action(_LimitBreakAction {});
        }
        static Action ReaperAction(int baseDamage) {
            return Action(_ReaperAction {baseDamage});
        }
        static Action RitualDaggerAction(int idx, int damage) {
            return Action(_RitualDaggerAction {idx, damage});
        }
        static Action SecondWindAction(int blockPerCard) {
            return Action(_SecondWindAction {blockPerCard});
        }
        static Action SeverSoulExhaustAction() {
            return Action(_SeverSoulExhaustAction {});
        }
        static Action SpotWeaknessAction(int target, int strength) {
            return Action(_SpotWeaknessAction {target, strength});
        }
        static Action WhirlwindAction(int baseDamage, int energy, bool useEnergy) {
            return Action(_WhirlwindAction {baseDamage, energy, useEnergy});
        }

        static Action AttackAllMonsterRecursive(DamageMatrix matrix, int timesRemaining) {
            return Action(_AttackAllMonsterRecursive {matrix, timesRemaining});
        }
    };

    bool clearOnCombatVictory(const Action &action);
    
    std::ostream& operator<<(std::ostream& os, const Action &action);
}

#endif //STS_LIGHTSPEED_ACTIONS_H
