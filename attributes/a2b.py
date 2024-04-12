import os
import numpy as np
from attributes.attributes_betas.a2b import A2B
from loguru import logger
logger.remove()

os.environ['PYOPENGL_PLATFORM'] = 'egl'

def data_A2S(ds_gender):

    rating_path = f'../samples/attributes/modeldata_for_a2s_{ds_gender}.pt'
    import joblib
    db = joblib.load(rating_path)
    # ['rating_label', 'ratings', 'ratings_raw', 'ids', 'heights', 'bust', 'hips', 'waist']
    
    if 'rating' not in db.keys():
        db['rating'] = db['ratings']

    db['height_gt'] = db['heights'].astype(np.float32)
    db['chest'] = db['bust'].astype(np.float32) / 100
    db['waist'] = db['waist'].astype(np.float32) / 100
    db['hips'] = db['hips'].astype(np.float32) / 100
    
    return db

def body_data():
    # linguistic shape attributes 
    rating = [1.8, 4.3, 3.9, 1.2, 1.3, 1.7, 1.8, 1.2, 1.5, 1.1, 1.8, 2.0, 1.5, 4.2, 1.5]
    chest = 130 #cm
    waist = 120
    hips = 130

    heights = 1.7 #m

    data = {}
    # data['rating_label'] = rating_label
    data['rating'] = np.array([rating])

    data['height_gt'] = np.array([heights], dtype=np.float32)

    data['chest'] = np.array([chest / 100], dtype=np.float32)
    data['waist'] = np.array([waist / 100], dtype=np.float32)
    data['hips'] = np.array([hips / 100], dtype=np.float32)
    
    data['ids'] = ['Ali Tate']

    return data

def main(ds_gender = 'male', model_gender = 'male', input_type = '04b_ahcwh2s'):

    checkpoint_path = f'../data/trained_models/a2b/caesar-{ds_gender}_smplx-{model_gender}-10betas/poynomial/{input_type}.yaml/last.ckpt'
    loaded_model = A2B.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # dataset = data_A2S(ds_gender)
    dataset = body_data()

    test_input, _ = loaded_model.create_input_feature_vec(dataset)
    test_input = loaded_model.to_whw2s(test_input, None) if loaded_model.whw2s_model else test_input
    prediction = loaded_model.a2b.predict(test_input).detach().cpu().numpy()

    for idx, betas in enumerate(prediction):
        model_name = dataset['ids'][idx]
        print(f'Predicted bestas for {model_name}')
        print(betas)
        print()
        print([float(f"{num:.1f}") for num in betas])
        print()



if __name__ == "__main__":
    main(ds_gender = 'female', model_gender = 'female', input_type = '04b_ahcwh2s')
    main(ds_gender = 'male', model_gender = 'male', input_type = '04b_ahcwh2s')
