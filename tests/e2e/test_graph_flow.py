import pytest
from unittest.mock import MagicMock, patch
from src.orchestrator.graph import build_graph
from src.orchestrator.actions import OrchestratorAction
from src.tools.base import ToolResult

@pytest.fixture
def mock_clients():
    # Patch registry where it is USED in executor.py
    with patch("src.orchestrator.nodes.orchestrator.Router") as MockRouter, \
         patch("src.orchestrator.nodes.executor.registry") as mock_registry:
        
        router = MockRouter.return_value
        
        yield {
            "router": router,
            "registry": mock_registry
        }

def test_graph_simple_flow(mock_clients):
    # Setup
    router = mock_clients["router"]
    registry = mock_clients["registry"]
    
    # Mock Tool
    mock_tool = MagicMock()
    mock_tool.run.return_value = ToolResult(output="Tool Output", cost=0.01)
    registry.get.return_value = mock_tool

    # Sequence:
    # 1. Orchestrator -> router returns tool="test_tool"
    # 2. Executor -> runs "test_tool"
    # 3. Aggregate -> continues
    # 4. Orchestrator -> router returns tool="finish"
    # 5. Executor -> runs "finish" (mocked to return is_terminal=True)
    # 6. Aggregate -> sees final_response -> done

    router.route.side_effect = [
        OrchestratorAction(reasoning="step1", tool="test_tool", parameters={"p": "v"}),
        OrchestratorAction(reasoning="done", tool="finish", parameters={"answer": "Final Answer"})
    ]
    
    # Mock registry behavior
    def get_tool(name):
        tool = MagicMock()
        if name == "test_tool":
            tool.run.return_value = ToolResult(output="Tool Output")
            tool.name = "test_tool"
        elif name == "finish":
            # Finish tool returns terminal result
            tool.run.return_value = ToolResult(output="Final Answer", is_terminal=True)
            tool.name = "finish"
        return tool
    
    registry.get.side_effect = get_tool
    
    app = build_graph()
    
    initial_state = {
        "query": "Hello",
        "preferences": {},
        "messages": [],
        "tool_results": [],
        "iteration": 0,
        "total_cost": 0.0
    }
    
    config = {"configurable": {"thread_id": "1"}}
    
    # Execute
    result = app.invoke(initial_state, config)
    
    # Verify
    assert result["final_response"] == "Final Answer"
    assert result["iteration"] >= 2
    assert router.route.call_count == 2
    # Check that test_tool was executed
    # We can check tool_results in the final state
    assert any(r["source"] == "test_tool" for r in result["tool_results"])

def test_graph_max_iterations(mock_clients):
    # Setup
    router = mock_clients["router"]
    registry = mock_clients["registry"]
    
    # Always return a looping action
    router.route.return_value = OrchestratorAction(
        reasoning="loop", tool="loop_tool", parameters={}
    )
    
    mock_tool = MagicMock()
    mock_tool.run.return_value = ToolResult(output="Loop")
    registry.get.return_value = mock_tool
    
    app = build_graph()
    
    initial_state = {
        "query": "Loop me",
        "preferences": {},
        "messages": [],
        "tool_results": [],
        "iteration": 0,
        "total_cost": 0.0
    }
    
    # We rely on aggregate node's MAX_ITERATIONS (50). 
    # To test quickly, we can patch MAX_ITERATIONS or run a few steps and assert it continues, 
    # but the test name implies checking the limit. 
    # Let's mock aggregate node's MAX_ITERATIONS to be small for this test.
    
    with patch("src.orchestrator.nodes.aggregate.MAX_ITERATIONS", 3):
        config = {"configurable": {"thread_id": "3"}, "recursion_limit": 10}
        result = app.invoke(initial_state, config)
        
        assert "Reached maximum iteration limit" in result["final_response"]
        assert result["iteration"] >= 3

