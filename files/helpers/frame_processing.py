import serpent.cv


HEART_COLORS = {
    "lefts": {
        (236, 0, 4): "RED",
        (255, 0, 4): "RED",
        (97, 117, 163): "SOUL",
        (63, 63, 63): "BLACK",
    },
    "rights": {
        (236, 0, 4): "RED",
        (72, 87, 121): "SOUL",
        (48, 48, 48): "BLACK",
        (255, 255, 255): "ETERNAL"
    }
}


def frame_to_hearts(game_frame, game):
    heart_positions = range(1, 13)
    heart_labels = [f"HUD_HEART_{position}" for position in heart_positions]

    hearts = list()

    for heart_label in heart_labels:
        heart = serpent.cv.extract_region_from_image(game_frame.frame, game.screen_regions[heart_label])

        left_heart_pixel = tuple(heart[3, 5, :])
        right_heart_pixel = tuple(heart[3, 17, :])
        unknown_heart_pixel = tuple(heart[9, 11, :])

        if unknown_heart_pixel == (230, 230, 230):
            return hearts

        hearts.append(HEART_COLORS["lefts"].get(left_heart_pixel))
        hearts.append(HEART_COLORS["rights"].get(right_heart_pixel))

    return hearts
