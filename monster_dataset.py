from pathlib import Path
from PIL import Image
from datastream import Dataset, Datastream
import tensorflow as tf
import numpy as np


def ImageDatastream(image_paths):
    return Datastream(
        Dataset.from_subscriptable(list(image_paths))
        .map(Image.open)
    )


pokemon_images = (
    set(Path("data/prepare_tfrecord").glob("pokemon-png*/*.png")) # / 10
)
dungeon_humanoids_images = (
    set(Path("data/prepare_tfrecord").glob("DungeonGameHumanoids64/*.png")) # / 100
)
tinyhero_images = (
    set(Path("data/prepare_tfrecord").glob("tinyhero*/*.png")) # / 10
)
enchanted_forest_images = (
    set(Path("data/prepare_tfrecord").glob("Enchanted_Forest/*.png")) # / 4
)

singles_images = [
    set(Path("data/prepare_tfrecord").glob("adventurer/*.png")),
    set(Path("data/prepare_tfrecord").glob("ArcherHero-*/*.png")),
    set(Path("data/prepare_tfrecord").glob("Droid_Zapper*/*.png")),
    set(Path("data/prepare_tfrecord").glob("Dude_Monster_Attack2_6/*.png")),
    set(Path("data/prepare_tfrecord").glob("FlyingEye/*.png")),
    set(Path("data/prepare_tfrecord").glob("ghost-idle_64x64/*.png")),
    set(Path("data/prepare_tfrecord").glob("HeavyBandit/*.png")),
    set(Path("data/prepare_tfrecord").glob("Knight-Idle-Sheet_64x64/*.png")),
    set(Path("data/prepare_tfrecord").glob("KnightIdle_strip/*.png")),
    set(Path("data/prepare_tfrecord").glob("LightBandit/*.png")),
    set(Path("data/prepare_tfrecord").glob("loreon_character/*.png")),
    set(Path("data/prepare_tfrecord").glob("MainCharacter_50x44/*.png")),
    set(Path("data/prepare_tfrecord").glob("Mud_Guard-Run_32x23/*.png")),
    set(Path("data/prepare_tfrecord").glob("Owlet_Monster_Attack2_6/*.png")),
    set(Path("data/prepare_tfrecord").glob("Pink_Monster_Attack2_6/*.png")),
    set(Path("data/prepare_tfrecord").glob("pokemon-character1_32x32/*.png")),
    set(Path("data/prepare_tfrecord").glob("pokemon-character2_32x32/*.png")),
    set(Path("data/prepare_tfrecord").glob("Skeleton_Mage_48x52/*.png")),
    set(Path("data/prepare_tfrecord").glob("Skeleton_Warrior_45x50/*.png")),
    set(Path("data/prepare_tfrecord").glob("spirit_boxer_run_32x44/*.png")),
    set(Path("data/prepare_tfrecord").glob("Sprites-Enemy01/*.png")),
    set(Path("data/prepare_tfrecord").glob("Sprites-Enemy02/*.png")),
    set(Path("data/prepare_tfrecord").glob("sprites_viking_axe/*.png")),
    set(Path("data/prepare_tfrecord").glob("warrior/*.png")),
    set(Path("data/prepare_tfrecord").glob("Warrior_Attack/*.png")),
    set(Path("data/prepare_tfrecord").glob("Warrior_Fall/*.png")),
    set(Path("data/prepare_tfrecord").glob("Warrior_Idle/*.png")),
    set(Path("data/prepare_tfrecord").glob("Warrior_Jump/*.png")),
    set(Path("data/prepare_tfrecord").glob("Warrior_Run/*.png")),
]


other_images = (
    set(
        list(Path("data/prepare_tfrecord").glob("*/*.png"))
    )
    - pokemon_images
    - dungeon_humanoids_images
    - tinyhero_images 
    - enchanted_forest_images
    - set().union(*singles_images)
)

print("pokemon_images:", len(pokemon_images))
print("dungeon_humanoids_images:", len(dungeon_humanoids_images))
print("tinyhero_images:", len(tinyhero_images))
print("enchanted_forest_images:", len(enchanted_forest_images))
print("singles_images:", sum([len(images) for images in singles_images]))
print("other_images:", len(other_images))

datastream = Datastream.merge([
    (ImageDatastream(pokemon_images), len(pokemon_images) // 10 // len(singles_images)),
    (ImageDatastream(dungeon_humanoids_images), len(dungeon_humanoids_images) // 20 // len(singles_images)),
    (ImageDatastream(tinyhero_images), len(tinyhero_images) // 20 // len(singles_images)),
    (ImageDatastream(enchanted_forest_images), len(enchanted_forest_images) // 4 // len(singles_images)),
    (ImageDatastream(other_images), len(other_images) // len(singles_images)),
    (Datastream.merge([
        (ImageDatastream(images), 1)
        for images in singles_images
    ]), 1),
])

it = iter(
    datastream
    .map(lambda image: tf.convert_to_tensor(np.array(image, dtype=np.uint8)))
    .map(lambda image: dict(image=image))
    .data_loader(collate_fn=list, num_workers=0, n_batches_per_epoch=1e30, batch_size=1)
)


def generator():
    while True:
        yield next(it)[0]


def MonsterDataset():
    return tf.data.Dataset.from_generator(
        generator,
        output_signature={
            "image": tf.TensorSpec(shape=(64, 64, 3), dtype=tf.uint8),
        },
    )