#!/usr/bin/env python3
"""
Test script for SIMA agent enhanced capabilities
"""
import sys
sys.path.insert(0, 'src')

def test_help():
    print("Testing SIMA agent help system...")
    from agent.conversational_agent import ConversationalMinecraftAgent
    
    agent = ConversationalMinecraftAgent(auto_focus=False)
    print("\n" + "="*60)
    print("TESTING ENHANCED HELP SYSTEM")
    print("="*60)
    agent._show_help()
    
    print("\n" + "="*60)  
    print("TESTING ENHANCED STATUS SYSTEM")
    print("="*60)
    agent._show_status()

if __name__ == "__main__":
    test_help()
