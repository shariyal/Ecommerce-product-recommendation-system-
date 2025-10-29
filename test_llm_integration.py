import pandas as pd
from recommendation_code import create_demo_hybrid_recommendations, get_llm_explanation, init_gemini

def test_llm_with_recommendations():
    """Test LLM explanations with actual recommendations"""
    from recommendation_code import load_and_clean_dataset
    
    # Load data
    data = load_and_clean_dataset('clean_data.csv')
    print(f"Loaded data: {len(data)} rows")
    
    # Create hybrid recommendations
    hybrid_recs = create_demo_hybrid_recommendations(data)
    
    if hybrid_recs.empty:
        print("❌ No hybrid recommendations created")
        return
    
    print(f"✅ Created {len(hybrid_recs)} hybrid recommendations")
    print(hybrid_recs[['Name', 'Rating']].head())
    
    # Initialize Gemini
    client_wrapper, model_name, error = init_gemini()
    if error:
        print(f"❌ Gemini init failed: {error}")
        # Test with fallback explanations
        print("\nTesting fallback explanations:")
        for idx, row in hybrid_recs.head(3).iterrows():
            explanation = get_llm_explanation(None, row['UserID'], row['Name'], "similar products and user preferences")
            print(f"Product: {row['Name'][:50]}...")
            print(f"Explanation: {explanation}\n")
    else:
        print(f"✅ Gemini initialized: {model_name}")
        # Test with real LLM
        print("\nTesting real LLM explanations:")
        for idx, row in hybrid_recs.head(2).iterrows():
            explanation = get_llm_explanation(client_wrapper, row['UserID'], row['Name'], "beauty products, high ratings")
            print(f"Product: {row['Name'][:50]}...")
            print(f"Explanation: {explanation}\n")

if __name__ == "__main__":
    test_llm_with_recommendations()