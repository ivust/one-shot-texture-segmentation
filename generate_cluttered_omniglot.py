import numpy as np
from skimage.transform import AffineTransform, warp, rescale, rotate, resize

def generate_cluttered_omniglot(
        omniglot_characters,
        textures,
        batch_size=1,
        n_characters=8):

    imgs = []
    for _ in range(batch_size):
        img = generate_multiple_characters(omniglot_characters, textures, n_characters)
        imgs.append(img)
    return np.stack(imgs)

def transform(img, angle=None, scale=None, translation=None):
    if angle is None:
        angle = np.random.randint(0, 360)
        
    if scale is None:
        scale = np.random.uniform(0.5, 2.0, size=2)
        
    if translation is None:
        translation = np.random.uniform(-100, 100, size=2)
        
    img_size = img.shape
        
    img = rotate(img, angle)
    img = rescale(img, scale)
    img = pad_min(img)
    img = resize(img, img_size)
    
    aft = AffineTransform(translation=translation)
    img = warp(img, aft)
    return img

def pad_min(img):
    s1, s2 = img.shape
    d = max(s1, s2) - min(s1, s2)
    
    if d == 0:
        return img
    elif s1 <= s2:
        img = np.pad(img, [(d // 2, d // 2), (0, 0)], 'constant')
    else:
        img = np.pad(img, [(0, 0), (d // 2, d // 2)], 'constant')
        
    return img

def generate_multiple_characters(all_characters, all_textures, n_characters):
    characters_idx = np.random.choice(all_characters.shape[0], n_characters)
    textures_idx = np.random.choice(all_textures.shape[0], n_characters)
    img_combined = np.zeros((256, 256, 3))
    for character_id, texture_id in zip(characters_idx, textures_idx):
        img = all_characters[character_id]
        img = np.pad(img, ((75, 76), (75, 76)), 'constant')
        img = transform(img) > 0.0
        img = np.expand_dims(img, -1)
        img_combined = img_combined * (1 - img) + img * all_textures[texture_id]
    return img_combined

def generate_validation_cluttered_omniglot(N=50):
    textures = np.load('validation_textures_conv1_1.npy')
    omniglot = np.load('validation_omniglot.npy')
    cluttered_omniglot = generate_cluttered_omniglot(omniglot, textures, batch_size=N)
    np.save('validation_cluttered_omniglot.npy', cluttered_omniglot)

if __name__ == '__main__':
    generate_validation_cluttered_omniglot()
