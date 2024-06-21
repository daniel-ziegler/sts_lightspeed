//
// Created by gamerpuppy on 7/4/2021.
//

#include "combat/Actions.h"
#include "combat/BattleContext.h"
#include "game/Game.h"

using namespace sts;

void _SetState::operator()(BattleContext& bc) const {
    bc.setState(state);
}

void _BuffPlayer::operator()(BattleContext &bc) const {
    if (s == PlayerStatus::CORRUPTION && !bc.player.hasStatus<PS::CORRUPTION>()) {
        bc.cards.onBuffCorruption();
    }
    bc.player.buff(s, amount);
}

void _DebuffPlayer::operator()(BattleContext &bc) const {
    bc.player.debuff(s, amount, isSourceMonster);
}

void _DecrementStatus::operator()(BattleContext &bc) const {
    bc.player.decrementStatus(s, amount);
}

void _RemoveStatus::operator()(BattleContext &bc) const {
    bc.player.setHasStatus(s, false);
}

void _BuffEnemy::operator()(BattleContext &bc) const {
    // todo check if alive?
    bc.monsters.arr[idx].buff(s, amount);
}

void _DebuffEnemy::operator()(BattleContext &bc) const {
    bc.debuffEnemy(s, idx, amount, isSourceMonster);
}

void _DebuffAllEnemy::operator()(BattleContext &bc) const {
	// todo this should just add all to bot immediately, not be called first
	// ^^ never mind i think adding to top is a workaround here
    for (int i = bc.monsters.monsterCount-1; i >= 0; --i) {
        if (bc.monsters.arr[i].isTargetable()) {
            bc.addToTop(Actions::DebuffEnemy(s, i, amount, isSourceMonster));
        }
    }
}

void BattleContext::debuffEnemy(MonsterStatus s, int idx, int amount, bool isSourceMonster) {
    // todo poison and snake skull
    Monster &e = monsters.arr[idx];

    if (e.hasStatus<MS::ARTIFACT>()) {
        e.decrementStatus<MS::ARTIFACT>();
        return;
    }

    e.addDebuff(s, amount, isSourceMonster);
    if (player.hasStatus<PS::SADISTIC>()) {
        addToBot(Actions::DamageEnemy(idx, player.getStatus<PS::SADISTIC>()));
    }

    if (s == MS::VULNERABLE && player.hasRelic<RelicId::CHAMPION_BELT>()) {
        addToBot(Actions::DebuffEnemy(MS::WEAK, idx, 1, false));
    }
}

void _AttackEnemy::operator()(BattleContext &bc) const {
    if (bc.monsters.arr[idx].isDeadOrEscaped()) {
        return;
    }

    bc.monsters.arr[idx].attacked(bc, damage);
    bc.checkCombat();
}

void _AttackAllEnemy::operator()(BattleContext &bc) const {
    // assume bc.curCard is the card being used

    int damageMatrix[5];
    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        if (!bc.monsters.arr[i].isDeadOrEscaped()) {
            damageMatrix[i] = bc.calculateCardDamage(bc.curCardQueueItem.card, i, baseDamage);
        }
    }

    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        if (!bc.monsters.arr[i].isDeadOrEscaped()) {
            bc.monsters.arr[i].attacked(bc, damageMatrix[i]);
        }
    }

    bc.checkCombat();
}

void _AttackAllEnemyMatrix::operator()(BattleContext &bc) const {
    // assume bc.curCard is the card being used

    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        if (!bc.monsters.arr[i].isDeadOrEscaped()) {
            bc.monsters.arr[i].attacked(bc, static_cast<int>(damageMatrix[i]));
        }
    }

    bc.checkCombat();
}

void _DamageEnemy::operator()(BattleContext &bc) const {
    if (bc.monsters.arr[idx].isDeadOrEscaped()) {
        return;
    }

    bc.monsters.arr[idx].damage(bc, damage);
    bc.checkCombat();
}

void _DamageAllEnemy::operator()(BattleContext& bc) const { // todo this is probably broken
    for (int idx = 0; idx < bc.monsters.monsterCount; idx++) {
        if (!bc.monsters.arr[idx].isDeadOrEscaped()) {
            bc.monsters.arr[idx].damage(bc, damage); // possible should addToBot here todo
        }
    }
    bc.checkCombat();
}

void _AttackPlayer::operator()(BattleContext &bc) const {
    bc.player.attacked(bc, idx, damage);
}

void _DamagePlayer::operator()(BattleContext &bc) const {
    bc.player.damage(bc, damage, selfDamage);
}

void _VampireAttack::operator()(BattleContext &bc) const {
    const auto mIdx = 0;
    auto &m = bc.monsters.arr[mIdx]; // only used by shelled parasite so idx is 0
    bc.player.attacked(bc, mIdx, damage);
    if (m.isAlive()) {
        m.heal(std::min(damage, static_cast<int>(bc.player.lastAttackUnblockedDamage)));
    }
}

void _PlayerLoseHp::operator()(BattleContext &bc) const {
    // TODO this doesn't take into account intangible or relics
    bc.player.loseHp(bc, hp, selfDamage);
}

void _HealPlayer::operator()(BattleContext &bc) const {
    bc.player.heal(amount);
}

void _MonsterGainBlock::operator()(BattleContext &bc) const {
    bc.monsters.arr[idx].block += amount;
}

void _RollMove::operator()(BattleContext &bc) const {
    Monster &m = bc.monsters.arr[monsterIdx];
    m.rollMove(bc);
}

void _ReactiveRollMove::operator()(BattleContext &bc) const {
    // writhing mass is always monster 0
    Monster &m = bc.monsters.arr[0];

    for (int i = 0 ; i < m.getStatus<MS::REACTIVE>(); ++i) {
        m.rollMove(bc);
    }
    m.setStatus<MS::REACTIVE>(0);
}

void _NoOpRollMove::operator()(BattleContext &bc) const {
    bc.noOpRollMove();
}

void _ChangeStance::operator()(BattleContext& bc) const {
    // TODO
}

void _GainEnergy::operator()(BattleContext &bc) const {
    bc.player.gainEnergy(amount);
}

void _GainBlock::operator()(BattleContext &bc) const {
    bc.player.gainBlock(bc, amount);
}
//
//Action Actions::GainBlockFromCard(int amount) {
//    return {[=] (BattleContext &bc) {
//
//        // TODO
//        bc.player.block += amount;
//    };
//}

void _DrawCards::operator()(BattleContext &bc) const {
    bc.drawCards(amount);
}

void _EmptyDeckShuffle::operator()(BattleContext &bc) const {
    java::Collections::shuffle(
            bc.cards.discardPile.begin(),
            bc.cards.discardPile.end(),
            java::Random(bc.shuffleRng.randomLong())
    );

    bc.cards.moveDiscardPileIntoToDrawPile();
}

void _ShuffleDrawPile::operator()(BattleContext &bc) const {
    java::Collections::shuffle(
            bc.cards.drawPile.begin(),
            bc.cards.drawPile.end(),
            java::Random(bc.shuffleRng.randomLong())
    );
}

void _ShuffleTempCardIntoDrawPile::operator()(BattleContext &bc) const {
    CardInstance c(id);
    for (int i = 0; i < count; ++i) {
        const int idx = bc.cards.drawPile.empty() ? 0 : bc.cardRandomRng.random(static_cast<int>(bc.cards.drawPile.size()-1));
        bc.cards.createTempCardInDrawPile(idx, c);
    }
}

void _PlayTopCard::operator()(BattleContext &bc) const {
    bc.playTopCardInDrawPile(monsterTargetIdx, exhausts);
}

void _MakeTempCardInHand::operator()(BattleContext &bc) const {
    // todo master reality when the action is created
    for (int i = 0; i < amount; ++i) {
        CardInstance c(card);
        c.uniqueId = bc.cards.nextUniqueCardId++;
        bc.cards.notifyAddCardToCombat(c);
        bc.moveToHandHelper(c);
    }
}

void _MakeTempCardInDrawPile::operator()(BattleContext &bc) const {
    // the random calculation is done in an effect so it be wrong to do it here?
    for (int i = 0; i < amount; ++i) {
        if (shuffleInto) {
            const int idx = bc.cards.drawPile.empty() ? 0 : bc.cardRandomRng.random(static_cast<int>(bc.cards.drawPile.size()-1));
            bc.cards.createTempCardInDrawPile(idx, c);
        }
        // todo else
    }
}

void _MakeTempCardInDiscard::operator()(BattleContext &bc) const {
    CardInstance c_copy(c);
    for (int i = 0; i < amount; ++i) {
        bc.cards.createTempCardInDiscard(c_copy);
    }
}

void _MakeTempCardsInHand::operator()(BattleContext &bc) const {
    for (auto c : cards) {
        c.uniqueId = bc.cards.nextUniqueCardId++;
        bc.cards.notifyAddCardToCombat(c);
        bc.moveToHandHelper(c);
    }
}


void _DiscardNoTriggerCard::operator()(BattleContext &bc) const {
    const auto &c = bc.curCardQueueItem.card;
    bc.cards.notifyRemoveFromHand(c);
    bc.cards.moveToDiscardPile(c);
}

void _ClearCardQueue::operator()(BattleContext &bc) const {
    bc.cardQueue.clear();
}

void _DiscardAtEndOfTurn::operator()(BattleContext &bc) const {
    bc.discardAtEndOfTurn();
}

void _DiscardAtEndOfTurnHelper::operator()(BattleContext &bc) const {
        bc.discardAtEndOfTurnHelper();
}

void _RestoreRetainedCards::operator()(BattleContext &bc) const {
        bc.restoreRetainedCards(count);
}

void _UnnamedEndOfTurnAction::operator()(BattleContext &bc) const {
    // EndTurnAction does this:
    //        AbstractDungeon.player.resetControllerValues();
    //        this.turnHasEnded = true;
    //        playerHpLastTurn = AbstractDungeon.player.currentHealth;

    bc.turnHasEnded = true;
    if (!bc.skipMonsterTurn) {
        bc.addToBot(Actions::MonsterStartTurnAction());
        bc.monsterTurnIdx = 0; // monstergroup preincrements this
    }
}

void _MonsterStartTurnAction::operator()(BattleContext &bc) const {
    bc.monsters.applyPreTurnLogic(bc);
}

void _TriggerEndOfTurnOrbsAction::operator()(BattleContext &bc) const {
    // todo
}

void _ExhaustTopCardInHand::operator()(BattleContext &bc) const {
    bc.exhaustTopCardInHand();
}

void _ExhaustSpecificCardInHand::operator()(BattleContext &bc) const {
    bc.exhaustSpecificCardInHand(idx, uniqueId);
}

void _DamageRandomEnemy::operator()(BattleContext &bc) const {
    const int idx = bc.monsters.getRandomMonsterIdx(bc.cardRandomRng, true);
    if (idx == -1) {
        return;
    }
    bc.monsters.arr[idx].damage(bc, damage);
    bc.checkCombat();
}

void _ExhaustRandomCardInHand::operator()(BattleContext &bc) const {
    for (int i = 0; i < count; ++i) {
        if (bc.cards.cardsInHand <= 0) {
            return;
        }
        const auto idx = bc.cards.getRandomCardIdxInHand(bc.cardRandomRng);
        auto c = bc.cards.hand[idx];
        bc.cards.removeFromHandAtIdx(idx);
        bc.triggerAndMoveToExhaustPile(c);
    }
}

void _MadnessAction::operator()(BattleContext &bc) const {

    bool haveNonZeroCost = false;
    bool haveNonZeroTurnCost = false;
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        const auto &c = bc.cards.hand[i];

        if (c.costForTurn > 0) {
            haveNonZeroTurnCost = true;
            break;
        }

        if (c.cost > 0) {
            haveNonZeroCost = true;
        }
    }

    const auto haveValidCard = haveNonZeroCost || haveNonZeroTurnCost;
    if (!haveValidCard) {
        return;
    }

    // always have 1 or more cards in hand here
//#pragma clang diagnostic push
//#pragma ide diagnostic ignored "EndlessLoop"
    while (true) {
        const auto randomIdx = bc.cardRandomRng.random(bc.cards.cardsInHand-1);
        auto &c = bc.cards.hand[randomIdx];

        if (haveNonZeroTurnCost) {
            if (c.costForTurn > 0) {
                c.cost = 0;
                c.costForTurn = 0;
                break;
            } else {
                continue;
            }

        } else {
            if (c.cost > 0) {
                c.cost = 0;
                c.costForTurn = 0;
                break;

            } else {
                continue;
            }
        }
    }
//#pragma clang diagnostic pop

}

void _RandomizeHandCost::operator()(BattleContext &bc) const {
        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            auto &c = bc.cards.hand[i];
            if (c.cost >= 0) {
                int newCost = bc.cardRandomRng.random(3);
                c.cost = newCost;
                c.costForTurn = newCost;
            }
        }
}

void _GainBlockRandomEnemy::operator()(BattleContext &bc) const {
        int validIdxs[5];
        int validCount = 0;

        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            const auto &m = bc.monsters.arr[i];
            if (i != sourceMonster && !m.isDying()) {
                validIdxs[validCount++] = i;
            }
        }

        int targetIdx;
        if (validCount > 0) {
            targetIdx = validIdxs[bc.aiRng.random(validCount - 1)];
        } else {
            targetIdx = sourceMonster;
        }

        bc.monsters.arr[targetIdx].addBlock(amount);
}

void _SummonGremlins::operator()(BattleContext &bc) const {
    // gremlin leader searches in the order 1, 2, 0 for open space
    int openIdxCount = 0;
    int newGremlinIdxs[2];
    if (bc.monsters.arr[1].isDying()) {
        newGremlinIdxs[openIdxCount++] = 1;
    }
    if (bc.monsters.arr[2].isDying()) {
        newGremlinIdxs[openIdxCount++] = 2;
    }
    if (openIdxCount < 2 && bc.monsters.arr[0].isDying()) {
        newGremlinIdxs[openIdxCount++] = 0;
    }
#ifdef sts_asserts
    assert(openIdxCount == 2);
#endif

    auto &gremlin0 = bc.monsters.arr[newGremlinIdxs[0]];
    auto &gremlin1 = bc.monsters.arr[newGremlinIdxs[1]];

    gremlin0 = Monster();
    gremlin1 = Monster();

    gremlin0.construct(bc, MonsterGroup::getGremlin(bc.aiRng), newGremlinIdxs[0]);
    gremlin1.construct(bc, MonsterGroup::getGremlin(bc.aiRng), newGremlinIdxs[1]);
    bc.monsters.monstersAlive += 2;

    if (bc.player.hasRelic<R::PHILOSOPHERS_STONE>()) {
        gremlin0.buff<MS::STRENGTH>(1);
        gremlin1.buff<MS::STRENGTH>(1);
    }
    gremlin0.buff<MS::MINION>();
    gremlin1.buff<MS::MINION>();

    gremlin0.rollMove(bc);
    gremlin1.rollMove(bc);
}

void _SpawnTorchHeads::operator()(BattleContext &bc) const {
    const auto spawnCount = 3-bc.monsters.monstersAlive;
#ifdef sts_asserts
    assert(spawnCount > 0);
#endif
    const int spawnIdxs[2] {(bc.monsters.arr[1].isDying() ? 1 : 0), 0};

    for (int i = 0; i < spawnCount; ++i) {
        const auto idx = spawnIdxs[i];
        auto &torchHead = bc.monsters.arr[idx];
        torchHead = Monster();
        torchHead.construct(bc, MonsterId::TORCH_HEAD, idx);
        torchHead.initHp(bc.monsterHpRng, bc.ascension); // bug somewhere in game
        torchHead.setMove(MMID::TORCH_HEAD_TACKLE);
        torchHead.buff<MS::MINION>();

        if (bc.player.hasRelic<R::PHILOSOPHERS_STONE>()) {
            torchHead.buff<MS::STRENGTH>(1);
        }
        ++bc.monsters.monstersAlive;
    }

    for (int i = 0; i < spawnCount; ++i) {
        bc.noOpRollMove();
    }
}

void _SpireShieldDebuff::operator()(BattleContext &bc) const {
    if (bc.aiRng.randomBoolean()) {
        bc.player.debuff<PS::FOCUS>(-1);
    } else {
        bc.player.debuff<PS::STRENGTH>(-1);
    }
}


void _OnAfterCardUsed::operator()(BattleContext &bc) const {
    bc.onAfterUseCard();
}

void _PutRandomCardsInDrawPile::operator()(BattleContext &bc) const {
    CardId ids[5];
    for (int i = 0; i < count; ++i) {
        ids[i] = getTrulyRandomCardInCombat(bc.cardRandomRng, bc.player.cc, type);
    }

    for (int i = 0; i < count; ++i) {
        CardInstance card(ids[i], false);
        card.cost = 0;
        card.costForTurn = 0;

        const int idx = bc.cards.drawPile.empty() ? 0 : bc.cardRandomRng.random(static_cast<int>(bc.cards.drawPile.size()-1));
        bc.cards.createTempCardInDrawPile(idx, card);
    }
}

void _DiscoveryAction::operator()(BattleContext &bc) const {
    bc.haveUsedDiscoveryAction = true;
    bc.openDiscoveryScreen(sts::generateDiscoveryCards(bc.cardRandomRng, bc.player.cc, type), amount);
}

void _InfernalBladeAction::operator()(BattleContext &bc) const {
    const auto cid = getTrulyRandomCardInCombat(bc.cardRandomRng, bc.player.cc, CardType::ATTACK);
    CardInstance c(cid);
    c.setCostForTurn(0);
    bc.addToTop( Actions::MakeTempCardInHand(c) );
}

void _JackOfAllTradesAction::operator()(BattleContext &bc) const {
    const auto c1 = sts::getTrulyRandomColorlessCardInCombat(bc.cardRandomRng);
    bc.addToTop( Actions::MakeTempCardInHand(c1) );
    if (upgraded) {
        auto c2 = sts::getTrulyRandomColorlessCardInCombat(bc.cardRandomRng);
        bc.addToTop( Actions::MakeTempCardInHand(c2) );
    }
}

void _TransmutationAction::operator()(BattleContext &bc) const {
    const auto effectAmount = energy + (bc.player.hasRelic<R::CHEMICAL_X>() ? 2 : 0);

    if (effectAmount == 0) {
        return;
    }

    std::vector<CardInstance> cards;
   for (int i = 0; i < effectAmount; ++i) {
       const auto cid = sts::getTrulyRandomColorlessCardInCombat(bc.cardRandomRng);
       CardInstance c(cid, upgraded);
       c.setCostForTurn(0);
       cards.push_back(c);
   }
   bc.addToBot( Actions::MakeTempCardsInHand(cards) );

   if (useEnergy) {
       bc.player.useEnergy(bc.player.energy);
   }
}

void _ViolenceAction::operator()(BattleContext &bc) const {
    // todo a faster algorithm for inserting into the attack list
    fixed_list<int,CardManager::MAX_GROUP_SIZE> attackIdxList;
    for (int i = 0; i < bc.cards.drawPile.size(); ++i) {
        const auto &c = bc.cards.drawPile[i];
        if (c.getType() == CardType::ATTACK) {

            if (attackIdxList.empty()) {
                attackIdxList.push_back(i);
            } else {
                const auto randomIdx = bc.cardRandomRng.random(attackIdxList.size() - 1);
                attackIdxList.insert(randomIdx, i);
            }
        }
    }

    if (attackIdxList.empty()) {
        return;
    }

    int removeIdxs[4];
    // hack to do this faster: the attackList is just pushed forward by i so we skip removing from bottom
    int i = 0;
    for (; i < count; ++i) {
        if (attackIdxList.size()-i <= 0) {
            return;
        }

        java::Collections::shuffle(attackIdxList.begin()+i, attackIdxList.end(), java::Random(bc.shuffleRng.randomLong()));
        const auto removeIdx = attackIdxList[i];
        removeIdxs[i] = removeIdx;

        const auto &c = bc.cards.drawPile[removeIdx];
        if (bc.cards.cardsInHand == 10) {
            bc.cards.moveToDiscardPile(c);
        } else {
            bc.cards.moveToHand(c);
        }
    }

    std::sort(removeIdxs, removeIdxs+i);
    for (int x = i-1; x >= 0; --x) {
        const auto drawPileRemoveIdx = removeIdxs[x];
        bc.cards.removeFromDrawPileAtIdx(drawPileRemoveIdx);
    }

}

// todo the amount should be the copies put into the hand 2 if have sacred bark and liquid memories
void _BetterDiscardPileToHandAction::operator()(BattleContext &bc) const {
    if (bc.cards.discardPile.empty()) {
        return;
    }
    if (bc.cards.discardPile.size() == 1) {
        const int idx = 0;
        bc.chooseDiscardToHandCard(0, task==CardSelectTask::LIQUID_MEMORIES_POTION);
    } else {
        bc.openSimpleCardSelectScreen(task, 1);
    }
}

void _ArmamentsAction::operator()(BattleContext &bc) const {
    int canUpgradeCount = 0;
    int lastUpgradeIdx = 0;
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        if (bc.cards.hand[i].canUpgrade()) {
            ++canUpgradeCount;
            lastUpgradeIdx = i;
        }
    }

    if (canUpgradeCount == 0) {
        // do nothing

    } else if (canUpgradeCount == 1) {
        bc.cards.hand[lastUpgradeIdx].upgrade();

    } else {
        bc.openSimpleCardSelectScreen(CardSelectTask::ARMAMENTS, 1);
    }
}

void _DualWieldAction::operator()(BattleContext &bc) const {

//        fixed_list<CardInstance, 10> validCards;
//        fixed_list<CardInstance, 10> invalidCards;
//
//        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
//            const bool valid = bc.cards.hand[i].getType() == CardType::ATTACK || bc.cards.hand[i].getType() == CardType::POWER;
//            if (valid) {
//                validCards.push_back(bc.cards.hand[i]);
//
//            } else {
//                invalidCards.push_back(bc.cards.hand[i]);
//            }
//        }

    int validCount = 0;
    int lastValidIdx = 0;

    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        const bool valid = bc.cards.hand[i].getType() == CardType::ATTACK || bc.cards.hand[i].getType() == CardType::POWER;
        if (valid) {
            ++validCount;
            lastValidIdx = i;
        }
    }

    if (validCount == 0) {
        return;
    }

    if (validCount == 1) {
        for (int i = 0; i < copyCount; ++i) {
            auto c = bc.cards.hand[lastValidIdx];
            if (bc.cards.cardsInHand + 1 <= CardManager::MAX_HAND_SIZE) {
                bc.cards.createTempCardInHand(c);

            } else {
                bc.cards.createTempCardInDiscard(c);

            }
        }

    } else {
        bc.inputState = InputState::CARD_SELECT;
        bc.cardSelectInfo.cardSelectTask = CardSelectTask::DUAL_WIELD;
        bc.cardSelectInfo.dualWield_CopyCount() = copyCount;

    }


}

void _ExhumeAction::operator()(BattleContext &bc) const {
    // todo this is bugged because the selected card cannot be exhume
    if (bc.cards.exhaustPile.empty() || bc.cards.cardsInHand == 10) {
        return;
    }

    int nonExhumeCards = 0;
    int lastNonExhumeIdx = -1;
    for (int i = 0; i < bc.cards.exhaustPile.size(); ++i) {
        if (bc.cards.exhaustPile[i].id != CardId::EXHUME) {
            ++nonExhumeCards;
            lastNonExhumeIdx = i;
        }
    }

    if (nonExhumeCards == 0) {
        return;

    } else if (nonExhumeCards == 1) {
        bc.chooseExhumeCard(lastNonExhumeIdx);

    } else {
        bc.cardSelectInfo.cardSelectTask = CardSelectTask::EXHUME;
        bc.inputState = InputState::CARD_SELECT;
    }
}

void _ForethoughtAction::operator()(BattleContext &bc) const {
    if (bc.cards.cardsInHand == 0) {
        return;
    }

    // todo implement Upgraded version
//        //
//        if (upgraded) {
//            bc.cardSelectInfo.cardSelectTask = CardSelectTask::FORETHOUGHT;
//            bc.cardSelectInfo.canPickAnyNumber = true;
//            bc.inputState = InputState::CARD_SELECT;
//
//        } else {
    if (bc.cards.cardsInHand == 1) {
        bc.chooseForethoughtCard(0);
    } else {
        bc.cardSelectInfo.cardSelectTask = CardSelectTask::FORETHOUGHT;
        bc.cardSelectInfo.canPickAnyNumber = false;
        bc.inputState = InputState::CARD_SELECT;
    }
//        }
}

void _HeadbuttAction::operator()(BattleContext &bc) const {
    if (bc.cards.discardPile.empty()) {
        return;

    } else if (bc.cards.discardPile.size() == 1) {
        bc.chooseHeadbuttCard(0);
    } else {
        bc.openSimpleCardSelectScreen(CardSelectTask::HEADBUTT, 1);
    }
}

void _ChooseExhaustOne::operator()(BattleContext &bc) const {
    if (bc.cards.cardsInHand == 0) {
        return;

    } else if (bc.cards.cardsInHand == 1) {
        bc.chooseExhaustOneCard(0);

    } else {
        bc.openSimpleCardSelectScreen(CardSelectTask::EXHAUST_ONE, 1);

    }
}

void _DrawToHandAction::operator()(BattleContext &bc) const {
    int count = 0;
    int idx = 0;

    for (int i = 0; i < bc.cards.drawPile.size(); ++i) {
        const auto &c = bc.cards.drawPile[i];
        if (c.getType() == cardType) {
            if (count > 0) {
                // for keeping rng consistent with game
                // the game creates a temporary list with the skills
                bc.cardRandomRng.random(count - 1);
            }
            idx = i;
            ++count;
        }
    }

    if (count == 0) {
        return;
    }

    if (count == 1) {
        bc.chooseDrawToHandCards(&idx, 1);

    } else {
        bc.cardSelectInfo.cardSelectTask = task;
        bc.inputState = InputState::CARD_SELECT;
    }

}

void _WarcryAction::operator()(BattleContext &bc) const {
    // todo if the handSize equals or less than the cardsToChoose just choose them here
    if (bc.cards.cardsInHand == 0) {
        return;
    }

    if (bc.cards.cardsInHand == 1) {
        bc.cardRandomRng.random(1);
        bc.chooseWarcryCard(0);

    } else {
        bc.inputState = InputState::CARD_SELECT;
        bc.cardSelectInfo.cardSelectTask = CardSelectTask::WARCRY;
    }
}

void _TimeEaterPlayCardQueueItem::operator()(BattleContext &bc) const {
    auto item = this->item;
    item.exhaustOnUse |= bc.curCardQueueItem.card.doesExhaust();
    item.triggerOnUse = false;
    bc.curCardQueueItem = item;
    bc.onAfterUseCard();
}

void _UpgradeAllCardsInHand::operator()(BattleContext &bc) const {
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        bc.cards.hand[i].upgrade();
    }
}

void _EssenceOfDarkness::operator()(BattleContext &bc) const {
}

void _IncreaseOrbSlots::operator()(BattleContext &bc) const {
}

void _SuicideAction::operator()(BattleContext &bc) const {
    auto &m = bc.monsters.arr[monsterIdx];
    if (triggerRelics) {
        if (m.isAlive()) {
            m.damage(bc, m.curHp);
        }
    } else {
        bc.monsters.arr[monsterIdx].suicideAction(bc);
    }
}

void _PoisonLoseHpAction::operator()(BattleContext &bc) const {
}

void _RemovePlayerDebuffs::operator()(BattleContext &bc) const {
    bc.player.removeDebuffs();
}

void _UpgradeRandomCardAction::operator()(BattleContext &bc) const {
    fixed_list<int,10> upgradeableHandIdxs;
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        if (bc.cards.hand[i].canUpgrade()) {
            upgradeableHandIdxs.push_back(i);
        }
    }

    if (upgradeableHandIdxs.empty()) {
        return;
    }

    java::Collections::shuffle(
            upgradeableHandIdxs.begin(),
            upgradeableHandIdxs.end(),
            java::Random(bc.shuffleRng.randomLong())
    );

    const auto upgradeIdx = upgradeableHandIdxs[0];
    bc.cards.hand[upgradeIdx].upgrade();
}

void _CodexAction::operator()(BattleContext &bc) const {
    bc.inputState = InputState::CARD_SELECT;
    bc.cardSelectInfo.cardSelectTask = CardSelectTask::CODEX;
    bc.cardSelectInfo.codexCards() =
            generateDiscoveryCards(bc.cardRandomRng, CharacterClass::IRONCLAD, CardType::INVALID);
}

void _ExhaustMany::operator()(BattleContext &bc) const {
    bc.inputState = InputState::CARD_SELECT;
    bc.cardSelectInfo.cardSelectTask = CardSelectTask::EXHAUST_MANY;
    bc.cardSelectInfo.pickCount = limit;
}

void _GambleAction::operator()(BattleContext &bc) const {
    bc.inputState = InputState::CARD_SELECT;
    bc.cardSelectInfo.cardSelectTask = CardSelectTask::GAMBLE;
}

void _ToolboxAction::operator()(BattleContext &bc) const {
    bc.inputState = InputState::CARD_SELECT;
    bc.cardSelectInfo.cardSelectTask = CardSelectTask::DISCOVERY;
    bc.cardSelectInfo.discovery_CopyCount() = 1;
    bc.cardSelectInfo.discovery_Cards() =
            generateDiscoveryCards(bc.cardRandomRng, bc.player.cc, CardType::STATUS); // status is mapped to colorless
}

void _DualityAction::operator()(BattleContext &bc) const {
    bc.player.buff<PS::DEXTERITY>(1);
    bc.player.debuff<PS::LOSE_DEXTERITY>(1);
}

void _ApotheosisAction::operator()(BattleContext &bc) const {

    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        auto &c = bc.cards.hand[i];
        if (c.canUpgrade()) {
            c.upgrade();
        }
    }

    for (auto &c : bc.cards.drawPile) {
        if (c.canUpgrade()) {
            c.upgrade();
        }
    }

    for (auto &c : bc.cards.discardPile) {
        if (c.canUpgrade()) {
            c.upgrade();
        }
    }

    for (auto &c : bc.cards.exhaustPile) {
        if (c.canUpgrade()) {
            c.upgrade();
        }
    }
}

void _DropkickAction::operator()(BattleContext &bc) const {
// assume bc.curCard is the card being used(BattleContext &bc) {
    if (bc.monsters.arr[targetIdx].isTargetable() && bc.monsters.arr[targetIdx].hasStatus<MS::VULNERABLE>()) {
        bc.addToTop(Actions::DrawCards(1));
        bc.addToTop(Actions::GainEnergy(1));
    }

    const auto &c = bc.curCardQueueItem.card;
    const int damage = bc.calculateCardDamage(c, targetIdx, c.isUpgraded() ? 8 : 5);
    bc.addToTop(Actions::AttackEnemy(targetIdx, damage));
}

void _EnlightenmentAction::operator()(BattleContext &bc) const {
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        auto &c = bc.cards.hand[i];
        if (c.costForTurn > 1) {
            c.costForTurn = 1;
        }
        if (upgraded && c.cost > 1) {
            c.costForTurn = 1;
            c.cost = 1;
        }
    }
}

void _EntrenchAction::operator()(BattleContext &bc) const {
    bc.player.gainBlock(bc, bc.player.block);
}

void _FeedAction::operator()(BattleContext &bc) const {
    auto &m = bc.monsters.arr[idx];
    if (m.isDeadOrEscaped()) {
        return;
    }
    bc.monsters.arr[idx].attacked(bc, damage);

    const bool effectTriggered = !m.hasStatus<MS::MINION>()
            && !m.isAlive()
            && !m.isHalfDead()
            && !(m.hasStatus<MS::REGROW>() && bc.monsters.monstersAlive > 0);

    if (effectTriggered) {
        bc.player.increaseMaxHp(upgraded ? 4 : 3);
    }

    bc.checkCombat();
}

void _FiendFireAction::operator()(BattleContext &bc) const {
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        bc.addToTop(Actions::AttackEnemy(targetIdx, calculatedDamage));
    }

    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        bc.addToTop( Actions::ExhaustRandomCardInHand(1) );
    }
}

void _HandOfGreedAction::operator()(BattleContext &bc) const {
    auto &m = bc.monsters.arr[idx];
    if (m.isDeadOrEscaped()) {
        return;
    }
    bc.monsters.arr[idx].damage(bc, damage);

    const bool effectTriggered = !m.hasStatus<MS::MINION>()
            && !m.isAlive()
            && !m.isHalfDead()
            && !(m.hasStatus<MS::REGROW>() && bc.monsters.monstersAlive > 0);

    if (effectTriggered) {
        bc.player.gainGold(bc, upgraded ? 25 : 20);
    }

    bc.checkCombat();
}

void _LimitBreakAction::operator()(BattleContext &bc) const {
    bc.player.buff<PS::STRENGTH>(bc.player.getStatus<PS::STRENGTH>());
}

void _ReaperAction::operator()(BattleContext &bc) const {

    int healAmount = 0;
    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        auto &m = bc.monsters.arr[i];
        if (m.isDeadOrEscaped()) {
            continue;
        }
        int preDamageHp = m.curHp;
        m.attacked(bc, bc.calculateCardDamage(bc.curCardQueueItem.card, i, baseDamage));
        healAmount += preDamageHp-m.curHp;
    }

    if (healAmount > 0) {
        bc.addToBot( Actions::HealPlayer(healAmount) );
    }

    //if (AbstractDungeon.getCurrRoom().monsters.areMonstersBasicallyDead()) {
    //                AbstractDungeon.actionManager.clearPostCombatActions();
    //            }
}

void _RitualDaggerAction::operator()(BattleContext &bc) const {
    auto &m = bc.monsters.arr[idx];
    if (m.isDeadOrEscaped()) {
        return;
    }
    bc.monsters.arr[idx].attacked(bc, damage);

    const bool shouldUpgrade = !m.hasStatus<MS::MINION>()
                               && !m.isAlive()
                               && !(m.hasStatus<MS::REGROW>() && bc.monsters.monstersAlive > 0);
    if (shouldUpgrade) {
        auto &c = bc.curCardQueueItem.card;
        const auto upgradeAmt = c.isUpgraded() ? 5 : 3;

        if (bc.curCardQueueItem.purgeOnUse) {
            bc.cards.findAndUpgradeSpecialData(c.uniqueId, upgradeAmt);
        }
        c.specialData += upgradeAmt;
    }

    bc.checkCombat();
}

void _SecondWindAction::operator()(BattleContext &bc) const {
    int cardIdxsToExhaust[10];
    int toExhaustCount = 0;

    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        const auto &c = bc.cards.hand[i];
        if (c.getType() != CardType::ATTACK) {
            cardIdxsToExhaust[toExhaustCount++] = i;
            bc.addToTop( Actions::GainBlock(blockPerCard) );
        }
    }

    for (int i = 0; i < toExhaustCount; ++i) {
        const auto handIdx = cardIdxsToExhaust[i];
        const auto &c = bc.cards.hand[handIdx];

        bc.addToTop( Actions::ExhaustSpecificCardInHand(handIdx, c.uniqueId) );
    }
}

void _SeverSoulExhaustAction::operator()(BattleContext &bc) const {
    for (int i = bc.cards.cardsInHand-1; i >= 0; --i) {
        const auto &c = bc.cards.hand[i];
        if (c.getType() != CardType::ATTACK) {
            bc.addToBot( Actions::ExhaustSpecificCardInHand(i, c.getUniqueId()) );
        }
    }
}

void _SwordBoomerangAction::operator()(BattleContext &bc) const {
    // pretty hacky until I can figure out a better solution
    const static CardInstance swordBoomerang {CardId::SWORD_BOOMERANG};
    const auto idx = bc.monsters.getRandomMonsterIdx(bc.cardRandomRng, true);
    if (idx == -1) {
        return;
    }

    int damage = bc.calculateCardDamage(swordBoomerang, idx, baseDamage);
    bc.addToTop(Actions::AttackEnemy(idx, damage));
}

void _SpotWeaknessAction::operator()(BattleContext &bc) const {
    if (bc.monsters.arr[target].isAttacking()) {
        bc.player.buff<PS::STRENGTH>(strength);
    }
}

void _WhirlwindAction::operator()(BattleContext &bc) const {
    // assume bc.curCard is the card being used

    if (useEnergy) {
        bc.player.useEnergy(bc.player.energy);
    }

    DamageMatrix damageMatrix {0};
    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        if (!bc.monsters.arr[i].isDeadOrEscaped()) {
            const auto calcDamage = bc.calculateCardDamage(bc.curCardQueueItem.card, i, baseDamage);

            damageMatrix[i] = static_cast<std::uint16_t>( // fit damage into uint16
                std::min(
                    static_cast<int>(std::numeric_limits<std::uint16_t>::max()),
                    calcDamage
                )
            );
        }
    }

    const auto effectAmount = energy + (bc.player.hasRelic<R::CHEMICAL_X>() ? 2 : 0);
    if (effectAmount > 0) {
        _AttackAllMonsterRecursive { damageMatrix, effectAmount }(bc);
    }
}

void _AttackAllMonsterRecursive::operator()(BattleContext &bc) const {
    if (timesRemaining <= 0) {
        return;
    }

    _AttackAllEnemyMatrix { matrix }(bc);

    if (timesRemaining > 1) {
        bc.addToTop(Actions::AttackAllMonsterRecursive(matrix, timesRemaining-1)); // todo should this be to the top? test with
    }
}

bool Action::operator==(const Action& rhs) const {
    if (type != rhs.type) {
        return false;
    }
    switch (type) {
#define ACTIONTYPE_EQ(name, ...) case ActionType_##name: return std::memcmp(&variant_##name, &rhs.variant_##name, sizeof(_##name)) == 0;
        FOREACH_ACTIONTYPE(ACTIONTYPE_EQ)
        default:
            assert(false);
    }
}

void Action::operator()(BattleContext& bc) const {
    switch (type) {
#define ACTIONTYPE_CALL(name, ...) case ActionType_##name: variant_##name(bc); break;
        FOREACH_ACTIONTYPE(ACTIONTYPE_CALL)
    }
}
namespace sts {

    bool clearOnCombatVictory(const Action &action) {
        return !(action.type == ActionType_AttackPlayer ||
            action.type == ActionType_DamagePlayer ||
            action.type == ActionType_PlayerLoseHp ||
            action.type == ActionType_HealPlayer ||
            action.type == ActionType_GainBlock ||
            action.type == ActionType_OnAfterCardUsed ||
            action.type == ActionType_TimeEaterPlayCardQueueItem ||
            action.type == ActionType_OnAfterCardUsed);
    }

    std::ostream& operator<<(std::ostream& os, const Action &action) {
        os << action.type;
        if (action.type == ActionType_ExhaustSpecificCardInHand) {
            os << "(" << action.variant_ExhaustSpecificCardInHand.uniqueId << ")";
        }
        return os;
    }

}