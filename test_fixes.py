import traceback
from recommendation_code import load_and_clean_dataset, content_based_recommendations, collaborative_filtering_recommendations


def quick_test():
    data_path = 'clean_data.csv'
    try:
        data = load_and_clean_dataset(data_path)
        print('Data loaded successfully, shape:', data.shape)
        if data.empty:
            print('Data is empty')
            return
        # Test content-based
        sample_product = data['Name'].dropna().iloc[0]
        print('Testing content-based with product:', sample_product)
        recs = content_based_recommendations(data, sample_product, top_n=3)
        print('Content-based recs found:', 0 if recs is None else len(recs))
        print(recs.head() if not recs.empty else 'No CB recs')
        # Test collaborative using a sample user
        sample_user = int(data['ID'].dropna().iloc[0])
        print('Testing collaborative for user:', sample_user)
        recs_cf = collaborative_filtering_recommendations(data, sample_user, top_n=3)
        print('CF recs found:', 0 if recs_cf is None else len(recs_cf))
        print(recs_cf.head() if not recs_cf.empty else 'No CF recs')
    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    quick_test()
