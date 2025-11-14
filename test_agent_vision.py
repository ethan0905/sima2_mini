#!/usr/bin/env python3
"""
Test the enhanced SIMA agent vision capabilities
"""
import sys
sys.path.insert(0, 'src')

def test_agent_vision_questions():
    print("🔍 Testing SIMA Agent Vision-based Responses...")
    print("=" * 60)
    
    from agent.conversational_agent import ConversationalMinecraftAgent
    
    # Create agent with vision
    agent = ConversationalMinecraftAgent(auto_focus=False)
    
    if not agent.enable_vision:
        print("❌ Vision not enabled - cannot test")
        return
        
    print("✅ Agent created with vision enabled")
    
    # Test vision questions
    questions = [
        "What do you see?",
        "Is it night or day?", 
        "What am I holding?",
        "How much health do I have?",
        "Am I hungry?",
        "What's my situation?"
    ]
    
    print("\n" + "="*60)
    print("TESTING VISION-BASED RESPONSES")
    print("="*60)
    
    for question in questions:
        print(f"\n👤 User: {question}")
        try:
            response = agent._parse_basic_command(question)
            print(f"🤖 Agent: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
            
        print("-" * 40)

if __name__ == "__main__":
    test_agent_vision_questions()
