//
// Created by gamerpuppy on 7/3/2021.
//

#ifndef STS_LIGHTSPEED_GAME_H
#define STS_LIGHTSPEED_GAME_H

#include <string>

#include "constants/Relics.h"
#include "constants/Cards.h"
#include "constants/CardPools.h"
#include "constants/CharacterClasses.h"
#include "constants/Potions.h"
#include "constants/Misc.h"

#include "game/Random.h"
#include "game/RelicContainer.h"

namespace sts {

    static constexpr float COLORLESS_RARE_CHANCE = 0.30f;

    namespace SeedHelper {
        constexpr int SEED_BASE = 35;

        int getDigitValue(char c);
        std::string getString(std::uint64_t seed);
        std::uint64_t getLong(const std::string &seed);
    };

    CardId getAnyColorCard(Random &cardRng, CardRarity rarity);
    CardId getRandomClassCardOfTypeAndRarity(Random &cardRng, CharacterClass cc, CardType type, CardRarity rarity);
    CardId getRandomClassCardOfRarity(Random &rng, CharacterClass cc, CardRarity rarity);
    CardId getRandomColorlessCardNeow(Random &rng, CardRarity rarity);
    CardId getColorlessCardFromPool(Random &cardRng, CardRarity rarity);

    CardId getRandomCurse(Random &cardRng);
    CardId getRandomCurse(Random &rng, CardId exclude);

    CardId getTrulyRandomCard(Random &rng, CharacterClass cc);
    CardId returnTrulyRandomColorlessCardFromAvailable(Random &rng, CardId exclude);

    CardId getTrulyRandomColorlessCardInCombat(Random &rng);
    CardId getTrulyRandomCardInCombat(Random &rng, CharacterClass cc);
    CardId getTrulyRandomCardInCombat(Random &rng, CharacterClass cc, CardType type);

    std::array<CardId, 3> generateDiscoveryCards(Random &rng, CharacterClass cc, CardType type);

    RelicTier returnRandomRelicTier(Random &relicRng, int act);
    RelicTier returnRandomRelicTierElite(Random &relicRng);

    Potion returnRandomPotion(Random &rng, CharacterClass cc, bool limited=false);
    Potion returnRandomPotionOfRarity(Random &rng, PotionRarity rarity, CharacterClass cc, bool limited= false);
    Potion getRandomPotion(Random &rng, CharacterClass cc);

    RelicId getRandomFace(const RelicContainer &relics, Random &rng);
    CardId getStartCardForEvent(CharacterClass cc);
    ChestSize getRandomChestSize(Random &treasureRng);
    RelicTier getMatryoshkaRelicTier(Random &relicRng);

    float getUpgradedCardChance(int act, int ascension);

}


#endif //STS_LIGHTSPEED_GAME_H
