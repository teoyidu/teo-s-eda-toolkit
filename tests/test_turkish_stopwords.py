#!/usr/bin/env python3
"""
Test script to verify Turkish stopwords loading
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_turkish_stopwords():
    """Test Turkish stopwords loading"""
    print("🧪 Testing Turkish Stopwords Loading")
    print("="*50)
    
    try:
        from src.data_quality.processors.boilerplate_cleaner import _load_turkish_stopwords
        
        # Load Turkish stopwords
        print("Loading Turkish stopwords...")
        stopwords = _load_turkish_stopwords()
        
        print(f"✅ Successfully loaded {len(stopwords)} Turkish stopwords")
        
        # Show some sample stopwords
        print("\n📋 Sample Turkish stopwords:")
        sample_words = list(stopwords)[:20]
        for i, word in enumerate(sample_words, 1):
            print(f"  {i:2d}. {word}")
        
        # Test some common Turkish words
        test_words = ['acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz']
        print("\n🔍 Testing common Turkish words:")
        for word in test_words:
            is_stopword = word in stopwords
            print(f"  '{word}': {'✅ Stopword' if is_stopword else '❌ Not a stopword'}")
        
        # Test some non-stopwords
        test_non_stopwords = ['mahkeme', 'karar', 'sözleşme', 'taraflar', 'imzalanmıştır']
        print("\n🔍 Testing legal terms (should not be stopwords):")
        for word in test_non_stopwords:
            is_stopword = word in stopwords
            print(f"  '{word}': {'❌ Stopword (unexpected)' if is_stopword else '✅ Not a stopword (correct)'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Turkish stopwords: {e}")
        return False

def test_boilerplate_cleaner_initialization():
    """Test boilerplate cleaner initialization"""
    print("\n🧪 Testing Boilerplate Cleaner Initialization")
    print("="*50)
    
    try:
        from src.data_quality.processors.boilerplate_cleaner import TurkishBoilerplateCleanerProcessor
        
        # Test configuration
        config = {
            'boilerplate_columns': {
                'text': {
                    'remove_duplicates': True,
                    'remove_header_footer': True,
                    'similarity_threshold': 0.8
                }
            },
            'use_turkish_embeddings': True,
            'embedding_model': 'tfidf',  # Use TF-IDF for faster testing
            'similarity_threshold': 0.8,
            'remove_turkish_stopwords': True,
            'normalize_turkish_text': True,
            'use_legal_patterns': True
        }
        
        print("Initializing Turkish boilerplate cleaner...")
        cleaner = TurkishBoilerplateCleanerProcessor(config)
        
        print("✅ Turkish boilerplate cleaner initialized successfully")
        print(f"  Turkish stopwords loaded: {len(cleaner.turkish_stopwords)}")
        print(f"  Legal patterns loaded: {len(cleaner.turkish_legal_patterns)}")
        print(f"  Common patterns loaded: {len(cleaner.common_patterns)}")
        print(f"  Embedding model: {cleaner.embedding_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing boilerplate cleaner: {e}")
        return False

def test_simple_cleaning():
    """Test simple text cleaning"""
    print("\n🧪 Testing Simple Text Cleaning")
    print("="*50)
    
    try:
        from src.data_quality.processors.boilerplate_cleaner import TurkishBoilerplateCleanerProcessor
        import pandas as pd
        
        # Simple test data
        test_data = {
            'text': [
                "TÜRKİYE CUMHURİYETİ\nANKARA 1. ASLİYE HUKUK MAHKEMESİ\n\nSayfa 1 / 3\n\nBu sözleşme taraflar arasında imzalanmıştır.\n\nMahkeme kararı kesinleşmiştir.",
                "T.C.\nİSTANBUL 2. ASLİYE HUKUK MAHKEMESİ\n\nSayfa 1 / 2\n\nBu sözleşme, taraflar arasında imzalanmıştır.\n\nYargıtay kararı kesinleşmiştir."
            ]
        }
        
        df = pd.DataFrame(test_data)
        
        # Configuration
        config = {
            'boilerplate_columns': {
                'text': {
                    'remove_duplicates': False,  # Disable for this test
                    'remove_header_footer': True,
                    'similarity_threshold': 0.8
                }
            },
            'use_turkish_embeddings': True,
            'embedding_model': 'tfidf',
            'similarity_threshold': 0.8,
            'remove_turkish_stopwords': True,
            'normalize_turkish_text': True,
            'use_legal_patterns': True
        }
        
        print("Processing test data...")
        cleaner = TurkishBoilerplateCleanerProcessor(config)
        result_df = cleaner.process(df)
        
        print("✅ Text cleaning completed successfully")
        
        # Show results
        for i in range(len(df)):
            print(f"\nDocument {i+1}:")
            print(f"  Original: {len(df.iloc[i]['text'])} chars")
            print(f"  Cleaned:  {len(result_df.iloc[i]['text'])} chars")
            
            original_len = len(df.iloc[i]['text'])
            cleaned_len = len(result_df.iloc[i]['text'])
            if original_len > 0:
                reduction = ((original_len - cleaned_len) / original_len) * 100
                print(f"  Reduction: {reduction:.1f}%")
            
            print(f"  Sample cleaned text: {result_df.iloc[i]['text'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in simple cleaning test: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Turkish Boilerplate Cleaner Test Suite")
    print("="*60)
    
    tests = [
        ("Turkish Stopwords Loading", test_turkish_stopwords),
        ("Boilerplate Cleaner Initialization", test_boilerplate_cleaner_initialization),
        ("Simple Text Cleaning", test_simple_cleaning)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Turkish boilerplate cleaner is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main() 