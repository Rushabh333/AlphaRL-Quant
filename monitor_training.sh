#!/bin/bash
# Monitoring script for 1M training

echo "🔍 RL TRAINING MONITOR"
echo "====================="
echo ""

# Check if training is running
if pgrep -f "train_agent.py" > /dev/null; then
    echo "✅ Training is RUNNING"
    echo ""
    
    # Show recent logs
    echo "📊 Recent Training Logs:"
    echo "------------------------"
    tail -20 logs/training_1m.log | grep -E "(INFO|ep_rew_mean|timesteps)"
    echo ""
    
    # Show model checkpoints
    echo "💾 Model Checkpoints:"
    echo "--------------------"
    ls -lht models/checkpoints/ 2>/dev/null | head -5 || echo "No checkpoints yet"
    echo ""
    
    # Show progress estimate
    echo "⏱️  Progress Tracking:"
    echo "---------------------"
    if [ -f "logs/training_1m.log" ]; then
        CURRENT_STEPS=$(grep -o "total_timesteps.*[0-9]\+" logs/training_1m.log | tail -1 | grep -o "[0-9]\+")
        if [ -n "$CURRENT_STEPS" ]; then
            PERCENT=$((CURRENT_STEPS * 100 / 1000000))
            echo "Steps: $CURRENT_STEPS / 1,000,000 ($PERCENT%)"
        else
            echo "Starting up..."
        fi
    fi
    echo ""
    
    echo "💡 Commands:"
    echo "  Watch live: tail -f logs/training_1m.log"
    echo "  TensorBoard: tensorboard --logdir=./logs/tensorboard/"
    echo "  Stop training: pkill -f train_agent.py"
else
    echo "⚠️  Training is NOT running"
    echo ""
    echo "Check logs: tail -50 logs/training_1m.log"
fi

echo ""
echo "====================="
