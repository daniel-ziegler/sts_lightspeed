//
// Created by gamerpuppy on 7/4/2021.
//

#ifndef STS_LIGHTSPEED_ACTIONS_H
#define STS_LIGHTSPEED_ACTIONS_H

#include <utility>
#include <functional>
#include <iostream>
#include <cstdint>
#include <cstring>

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

#define FOREACH_ACTIONTYPE(X) \
    X(SetState, InputState,state) \
    X(BuffPlayer, PlayerStatus,s, int,amount) \
    X(DebuffPlayer, PlayerStatus,s, int,amount, bool,isSourceMonster) \
    X(DecrementStatus, PlayerStatus,s, int,amount) \
    X(RemoveStatus, PlayerStatus,s) \
    X(BuffEnemy, MonsterStatus,s, int,idx, int,amount) \
    X(DebuffEnemy, MonsterStatus,s, int,idx, int,amount, bool,isSourceMonster) \
    X(DebuffAllEnemy, MonsterStatus,s, int,amount, bool,isSourceMonster) \
    X(AttackEnemy, int,idx, int,damage) \
    X(AttackAllEnemy, int,baseDamage) \
    X(AttackAllEnemyMatrix, DamageMatrix,damageMatrix) \
    X(DamageEnemy, int,idx, int,damage) \
    X(DamageAllEnemy, int,damage) \
    X(AttackPlayer, int,idx, int,damage) \
    X(DamagePlayer, int,damage, bool,selfDamage) \
    X(VampireAttack, int,damage) \
    X(PlayerLoseHp, int,hp, bool,selfDamage) \
    X(HealPlayer, int,amount) \
    X(MonsterGainBlock, int,idx, int,amount) \
    X(RollMove, int,monsterIdx) \
    X(ReactiveRollMove) \
    X(NoOpRollMove) \
    X(ChangeStance, Stance,stance) \
    X(GainEnergy, int,amount) \
    X(GainBlock, int,amount) \
    X(DrawCards, int,amount) \
    X(EmptyDeckShuffle) \
    X(ShuffleDrawPile) \
    X(ShuffleTempCardIntoDrawPile, CardId,id, int,count) \
    X(PlayTopCard, int,monsterTargetIdx, bool,exhausts) \
    X(MakeTempCardInHand, CardInstance,card, int,amount) \
    X(MakeTempCardInDrawPile, CardInstance,c, int,amount, bool,shuffleInto) \
    X(MakeTempCardInDiscard, CardInstance,c, int,amount) \
    X(MakeTempCardsInHand, std::vector<CardInstance>,cards ) \
    X(DiscardNoTriggerCard) \
    X(ClearCardQueue) \
    X(DiscardAtEndOfTurn) \
    X(DiscardAtEndOfTurnHelper) \
    X(RestoreRetainedCards, int,count) \
    X(UnnamedEndOfTurnAction) \
    X(MonsterStartTurnAction) \
    X(TriggerEndOfTurnOrbsAction) \
    X(ExhaustTopCardInHand) \
    X(ExhaustSpecificCardInHand, int,idx, std::int16_t,uniqueId) \
    X(DamageRandomEnemy, int,damage) \
    X(GainBlockRandomEnemy, int,sourceMonster, int,amount) \
    X(SummonGremlins) \
    X(SpawnTorchHeads) \
    X(SpireShieldDebuff) \
    X(ExhaustRandomCardInHand, int,count) \
    X(MadnessAction) \
    X(RandomizeHandCost) \
    X(UpgradeRandomCardAction) \
    X(CodexAction) \
    X(ExhaustMany, int,limit) \
    X(GambleAction) \
    X(ToolboxAction) \
    X(FiendFireAction, int,targetIdx, int,calculatedDamage) \
    X(SwordBoomerangAction, int,baseDamage) \
    X(PutRandomCardsInDrawPile, CardType,type, int,count) \
    X(DiscoveryAction, CardType,type, int,amount) \
    X(InfernalBladeAction) \
    X(JackOfAllTradesAction, bool,upgraded) \
    X(TransmutationAction, bool,upgraded, int,energy, bool,useEnergy) \
    X(ViolenceAction, int,count) \
    X(BetterDiscardPileToHandAction, int,amount, CardSelectTask,task) \
    X(ArmamentsAction) \
    X(DualWieldAction, int,copyCount) \
    X(ExhumeAction) \
    X(ForethoughtAction, bool,upgraded) \
    X(HeadbuttAction) \
    X(ChooseExhaustOne) \
    X(DrawToHandAction, CardSelectTask,task, CardType,cardType) \
    X(WarcryAction) \
    X(TimeEaterPlayCardQueueItem, CardQueueItem,item) \
    X(UpgradeAllCardsInHand) \
    X(OnAfterCardUsed) \
    X(EssenceOfDarkness, int,darkOrbsPerSlot) \
    X(IncreaseOrbSlots, int,count) \
    X(SuicideAction, int,monsterIdx, bool,triggerRelics) \
    X(PoisonLoseHpAction) \
    X(RemovePlayerDebuffs) \
    X(DualityAction) \
    X(ApotheosisAction) \
    X(DropkickAction, int,targetIdx) \
    X(EnlightenmentAction, bool,upgraded) \
    X(EntrenchAction) \
    X(FeedAction, int,idx, int,damage, bool,upgraded) \
    X(HandOfGreedAction, int,idx, int,damage, bool,upgraded) \
    X(LimitBreakAction) \
    X(ReaperAction, int,baseDamage) \
    X(RitualDaggerAction, int,idx, int,damage) \
    X(SecondWindAction, int,blockPerCard) \
    X(SeverSoulExhaustAction) \
    X(SpotWeaknessAction, int,target, int,strength) \
    X(WhirlwindAction, int,baseDamage, int,energy, bool,useEnergy) \
    X(AttackAllMonsterRecursive, DamageMatrix,matrix, int,timesRemaining)

#define PP_NARG(...) \
         PP_NARG_(__VA_ARGS__,PP_RSEQ_N())
#define PP_NARG_(...) \
         PP_ARG_N(__VA_ARGS__)
#define PP_ARG_N( \
          _1, _2, _3, _4, _5, _6, _7, _8, _9,_10, \
         _11,_12,_13,_14,_15,_16,_17,_18,_19,_20, \
         _21,_22,_23,_24,_25,_26,_27,_28,_29,_30, \
         _31,_32,_33,_34,_35,_36,_37,_38,_39,_40, \
         _41,_42,_43,_44,_45,_46,_47,_48,_49,_50, \
         _51,_52,_53,_54,_55,_56,_57,_58,_59,_60, \
         _61,_62,_63,N,...) N
#define PP_RSEQ_N() \
         63,62,61,60,                   \
         59,58,57,56,55,54,53,52,51,50, \
         49,48,47,46,45,44,43,42,41,40, \
         39,38,37,36,35,34,33,32,31,30, \
         29,28,27,26,25,24,23,22,21,20, \
         19,18,17,16,15,14,13,12,11,10, \
         9,8,7,6,5,4,3,2,1,0

#define FOREACH2_0(F, LF)
#define FOREACH2_1(F, LF, x)
#define FOREACH2_2(F, LF, a, b) LF(a, b)
#define FOREACH2_4( F, LF, a, b, ...) F(a, b) FOREACH2_2( F, LF, __VA_ARGS__)
#define FOREACH2_6( F, LF, a, b, ...) F(a, b) FOREACH2_4( F, LF, __VA_ARGS__)
#define FOREACH2_8( F, LF, a, b, ...) F(a, b) FOREACH2_6( F, LF, __VA_ARGS__)
#define FOREACH2_10(F, LF, a, b, ...) F(a, b) FOREACH2_8( F, LF, __VA_ARGS__)
#define FOREACH2_12(F, LF, a, b, ...) F(a, b) FOREACH2_10(F, LF, __VA_ARGS__)

#define CONCATENATE(arg1, arg2) CONCATENATE1(arg1, arg2)
#define CONCATENATE1(arg1, arg2) arg1##arg2

#define FOREACH2(F, F2, ...) CONCATENATE(FOREACH2_, PP_NARG(__VA_ARGS__))(F, F2, __VA_ARGS__)

#define FIELD(fieldtype, name) fieldtype name;
#define ACTIONDATA_STRUCT(type, ...) struct _##type { FOREACH2(FIELD, FIELD, __VA_ARGS__) void operator()(BattleContext& bc) const; };
FOREACH_ACTIONTYPE(ACTIONDATA_STRUCT)
    
#define ACTIONTYPE_NAME(type, ...) ActionType_##type,
enum ActionType {
    ActionType_INVALID,
    FOREACH_ACTIONTYPE(ACTIONTYPE_NAME)
};
     
struct Action {
    ActionType type = ActionType_INVALID;
    union {
#define ACTIONTYPE_VARIANT(type, ...) _##type variant_##type;
        FOREACH_ACTIONTYPE(ACTIONTYPE_VARIANT)
    };
    
    Action() {}
    ~Action() {}
    
    Action(const Action& rhs) {
        *this = rhs;
    }
    Action& operator=(const Action& rhs) {
        std::memcpy(this, &rhs, sizeof(Action));
        return *this;
    }
    
    void operator()(BattleContext& bc) const;
    bool operator==(const Action& rhs) const;
};
    struct Actions {

#define PARAM(type, name) type name,
#define LASTPARAM(type, name) type name
#define ARG(type, name) name,
#define ACTIONTYPE_MAKE(actiontype, ...) \
    static Action actiontype(FOREACH2(PARAM, LASTPARAM, __VA_ARGS__)) { \
        Action action; action.type=ActionType_##actiontype; \
        action.variant_##actiontype=_##actiontype{FOREACH2(ARG, ARG, __VA_ARGS__)}; \
        return action; \
    }

FOREACH_ACTIONTYPE(ACTIONTYPE_MAKE)
    
        // some convenience overloads

        static Action BuffPlayer(PlayerStatus status) {
            return BuffPlayer(status, 1);
        }
        static Action DebuffPlayer(PlayerStatus status, int amount=1) {
            return DebuffPlayer(status, amount, true);
        }
        static Action DecrementStatus(PlayerStatus status) {
            return DecrementStatus(status, 1);
        }
        static Action BuffEnemy(MonsterStatus status, int idx) {
            return BuffEnemy(status, idx, 1);
        }
        static Action DebuffEnemy(MonsterStatus status, int idx, int amount) {
            return DebuffEnemy(status, idx, amount, true);
        }
        static Action DebuffAllEnemy(MonsterStatus status, int amount) {
            return DebuffAllEnemy(status, amount, true);
        }
        static Action DamagePlayer(int damage) {
            return DamagePlayer(damage, false);
        }
        static Action PlayerLoseHp(int hp) {
            return PlayerLoseHp(hp, false);
        }
        static Action MakeTempCardInHand(CardInstance& card) {
            return MakeTempCardInHand(card, 1);
        }
        static Action MakeTempCardInHand(CardId card, bool upgraded=false, int amount=1) {
            return MakeTempCardInHand(CardInstance(card, upgraded), amount);
        }
        static Action MakeTempCardInDiscard(const CardInstance &c) {
            return MakeTempCardInDiscard(c, 1);
        }
        static Action ShuffleTempCardIntoDrawPile(CardId id) {
            return ShuffleTempCardIntoDrawPile(id, 1);
        }
        
    };

    bool clearOnCombatVictory(const Action &action);
    
    std::ostream& operator<<(std::ostream& os, const Action &action);
}

#endif //STS_LIGHTSPEED_ACTIONS_H
