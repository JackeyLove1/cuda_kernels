"""agent 模块单元测试（unittest + mock，不发起真实 API 请求）。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import agent


def _restore_tool_specs(saved: list[agent.ToolSpec]) -> None:
    agent._TOOL_SPECS[:] = saved


class TestLoadEnvFromDotenv(unittest.TestCase):
    @patch("agent.load_dotenv")
    def test_calls_load_dotenv_for_existing_files(self, mock_ld: MagicMock) -> None:
        script_dotenv = MagicMock()
        script_dotenv.is_file.return_value = True
        cwd_dotenv = MagicMock()
        cwd_dotenv.is_file.return_value = False

        script_dir = MagicMock()
        script_dir.__truediv__ = MagicMock(return_value=script_dotenv)

        file_path = MagicMock()
        file_path.resolve.return_value.parent = script_dir

        cwd_root = MagicMock()
        cwd_root.__truediv__ = MagicMock(return_value=cwd_dotenv)

        def path_factory(arg: object) -> MagicMock:
            if arg == agent.__file__:
                return file_path
            raise AssertionError(f"unexpected Path({arg!r})")

        with patch("agent.Path") as mock_path_cls:
            mock_path_cls.side_effect = path_factory
            mock_path_cls.cwd = MagicMock(return_value=cwd_root)
            agent.load_env_from_dotenv()

        mock_ld.assert_called_once_with(script_dotenv, override=False)


class TestDefaultChatModel(unittest.TestCase):
    def test_fallback(self) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_AGENT_MODEL", "OPENAI_MODEL")}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(agent.default_chat_model(), "gpt-4o-mini")

    def test_openai_agent_model_wins(self) -> None:
        with patch.dict(os.environ, {"OPENAI_AGENT_MODEL": "m1", "OPENAI_MODEL": "m2"}):
            self.assertEqual(agent.default_chat_model(), "m1")

    def test_openai_model_fallback(self) -> None:
        with patch.dict(os.environ, {"OPENAI_MODEL": "m-x"}):
            os.environ.pop("OPENAI_AGENT_MODEL", None)
            self.assertEqual(agent.default_chat_model(), "m-x")


class TestDescriptionFromDocstring(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(agent._description_from_docstring(""), "")
        self.assertEqual(agent._description_from_docstring("   \n  "), "")

    def test_first_paragraph_stops_at_blank(self) -> None:
        d = "Line one.\n\nLine two."
        self.assertEqual(agent._description_from_docstring(d), "Line one.")

    def test_stops_at_args(self) -> None:
        d = "Summary.\n\nArgs:\n    x: y"
        self.assertEqual(agent._description_from_docstring(d), "Summary.")


class TestParseParamDescriptions(unittest.TestCase):
    def test_sphinx_param(self) -> None:
        doc = ":param foo: bar\n"
        self.assertEqual(agent._parse_param_descriptions(doc)["foo"], "bar")

    def test_google_args(self) -> None:
        doc = """One line.

Args:
    a: first
    b: second
"""
        p = agent._parse_param_descriptions(doc)
        self.assertEqual(p.get("a"), "first")
        self.assertEqual(p.get("b"), "second")


class TestOptionalHelpers(unittest.TestCase):
    def test_is_optional(self) -> None:
        self.assertTrue(agent._is_optional(int | None))
        self.assertFalse(agent._is_optional(int))

    def test_strip_optional(self) -> None:
        self.assertIs(agent._strip_optional(str | None), str)


class TestJsonTypeFor(unittest.TestCase):
    def test_primitives(self) -> None:
        self.assertEqual(agent._json_type_for(str), {"type": "string"})
        self.assertEqual(agent._json_type_for(int), {"type": "integer"})
        self.assertEqual(agent._json_type_for(float), {"type": "number"})
        self.assertEqual(agent._json_type_for(bool), {"type": "boolean"})

    def test_list_and_dict(self) -> None:
        self.assertEqual(
            agent._json_type_for(list[str]),
            {"type": "array", "items": {"type": "string"}},
        )
        self.assertEqual(
            agent._json_type_for(dict[str, int]),
            {"type": "object", "additionalProperties": True},
        )


class TestBuildParametersSchema(unittest.TestCase):
    def test_required_and_descriptions(self) -> None:
        def sample(a: int, b: str = "x") -> None:
            """T.

            Args:
                a: aa
                b: bb
            """

        schema = agent._build_parameters_schema(
            sample,
            {"a": "aa", "b": "bb"},
        )
        self.assertIn("a", schema["required"])
        self.assertNotIn("b", schema["required"])
        self.assertEqual(schema["properties"]["a"]["description"], "aa")


class TestToolDecorator(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = list(agent._TOOL_SPECS)

    def tearDown(self) -> None:
        _restore_tool_specs(self._saved)

    def test_registers_name_and_schema(self) -> None:
        fresh: list[agent.ToolSpec] = []
        with patch.object(agent, "_TOOL_SPECS", fresh):

            @agent.tool(name="renamed", description="D")
            def my_add(x: float, y: float) -> float:
                """Ignored body desc.

                Args:
                    x: ex
                    y: why
                """
                return x + y

            self.assertEqual(len(fresh), 1)
            sp = fresh[0]
            self.assertEqual(sp.name, "renamed")
            self.assertEqual(sp.description, "D")
            self.assertIn("x", sp.parameters["properties"])
            self.assertEqual(sp.parameters["properties"]["x"]["description"], "ex")
            self.assertEqual(my_add(1, 2), 3)


class TestOpenaiToolSchemas(unittest.TestCase):
    def test_contains_registered_tools(self) -> None:
        names = {s["function"]["name"] for s in agent.openai_tool_schemas()}
        self.assertIn("add", names)
        self.assertIn("echo_upper", names)
        self.assertIn("read_file", names)
        self.assertIn("write_file", names)
        self.assertIn("patch_file", names)


class TestFsPathResolution(unittest.TestCase):
    def test_path_outside_root_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AGENT_FS_ROOT": tmp}):
                with self.assertRaises(PermissionError):
                    agent._resolve_path_under_fs_root("/etc/passwd")


class TestFileTools(unittest.TestCase):
    def test_write_then_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AGENT_FS_ROOT": tmp}):
                wout = json.loads(
                    agent.run_tool(
                        "write_file",
                        json.dumps({"path": "sub/t.txt", "content": "hello"}),
                    ),
                )
                self.assertTrue(wout.get("ok"))
                self.assertEqual(wout.get("chars_written"), 5)
                content = agent.run_tool(
                    "read_file",
                    json.dumps({"path": "sub/t.txt"}),
                )
                self.assertEqual(content, "hello")

    def test_read_outside_root_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AGENT_FS_ROOT": tmp}):
                err = json.loads(
                    agent.run_tool("read_file", '{"path": "/nonexistent_outside_root"}'),
                )
                self.assertIn("error", err)

    def test_patch_file_unique_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AGENT_FS_ROOT": tmp}):
                agent.run_tool(
                    "write_file",
                    json.dumps({"path": "f.txt", "content": "aa bb aa"}),
                )
                out = json.loads(
                    agent.run_tool(
                        "patch_file",
                        json.dumps(
                            {
                                "path": "f.txt",
                                "old_string": "bb",
                                "new_string": "XX",
                            },
                        ),
                    ),
                )
                self.assertTrue(out.get("ok"))
                self.assertEqual(out.get("replacements"), 1)
                self.assertEqual(
                    agent.run_tool("read_file", json.dumps({"path": "f.txt"})),
                    "aa XX aa",
                )

    def test_patch_file_rejects_multiple_matches_without_replace_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AGENT_FS_ROOT": tmp}):
                agent.run_tool(
                    "write_file",
                    json.dumps({"path": "f.txt", "content": "aa bb aa"}),
                )
                err = json.loads(
                    agent.run_tool(
                        "patch_file",
                        json.dumps(
                            {
                                "path": "f.txt",
                                "old_string": "aa",
                                "new_string": "Z",
                            },
                        ),
                    ),
                )
                self.assertIn("error", err)

    def test_patch_file_replace_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AGENT_FS_ROOT": tmp}):
                agent.run_tool(
                    "write_file",
                    json.dumps({"path": "f.txt", "content": "aa bb aa"}),
                )
                out = json.loads(
                    agent.run_tool(
                        "patch_file",
                        json.dumps(
                            {
                                "path": "f.txt",
                                "old_string": "aa",
                                "new_string": "Z",
                                "replace_all": True,
                            },
                        ),
                    ),
                )
                self.assertTrue(out.get("ok"))
                self.assertEqual(out.get("replacements"), 2)
                self.assertEqual(
                    agent.run_tool("read_file", json.dumps({"path": "f.txt"})),
                    "Z bb Z",
                )


class TestRunTool(unittest.TestCase):
    def test_unknown_tool(self) -> None:
        out = agent.run_tool("nonexistent_tool_xyz", "{}")
        self.assertIn("error", json.loads(out))

    def test_add_and_echo(self) -> None:
        self.assertEqual(float(agent.run_tool("add", '{"a": 1, "b": 2}')), 3.0)
        self.assertEqual(agent.run_tool("echo_upper", '{"text": "ab"}'), "AB")

    def test_bad_json_arguments(self) -> None:
        out = json.loads(agent.run_tool("add", "{not json}"))
        self.assertIn("error", out)


class TestRunRegisteredTool(unittest.TestCase):
    def test_invokes_fn(self) -> None:
        spec = agent.ToolSpec(
            name="n",
            description="",
            fn=lambda x: x * 2,
            parameters={},
        )
        self.assertEqual(agent._run_registered_tool(spec, {"x": 21}), 42)


class TestAssistantMessageToDict(unittest.TestCase):
    def test_content_only(self) -> None:
        msg = SimpleNamespace(content="hello", tool_calls=None)
        d = agent._assistant_message_to_dict(msg)
        self.assertEqual(d["role"], "assistant")
        self.assertEqual(d["content"], "hello")
        self.assertNotIn("tool_calls", d)

    def test_with_tool_calls(self) -> None:
        fn = SimpleNamespace(name="add", arguments='{"a":1}')
        tc = SimpleNamespace(id="t1", function=fn)
        msg = SimpleNamespace(content="", tool_calls=[tc])
        d = agent._assistant_message_to_dict(msg)
        self.assertEqual(len(d["tool_calls"]), 1)
        self.assertEqual(d["tool_calls"][0]["id"], "t1")


class TestAgentLoop(unittest.TestCase):
    def test_one_tool_round_then_final_text(self) -> None:
        msg1 = SimpleNamespace(content=None, tool_calls=None)
        tc_fn = SimpleNamespace(name="add", arguments='{"a": 2, "b": 3}')
        tc = SimpleNamespace(id="call_1", function=tc_fn)
        msg1.tool_calls = [tc]

        msg2 = SimpleNamespace(content="  sum is five  ", tool_calls=None)

        client = MagicMock()
        client.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(message=msg1)]),
            SimpleNamespace(choices=[SimpleNamespace(message=msg2)]),
        ]

        out = agent.agent_loop(client, user_text="2+3", model="fake-model")
        self.assertEqual(out, "sum is five")

        second_msgs = client.chat.completions.create.call_args_list[1].kwargs["messages"]
        self.assertTrue(any(m.get("role") == "tool" for m in second_msgs))

    def test_max_turns_message(self) -> None:
        msg = SimpleNamespace(content=None, tool_calls=None)
        tc_fn = SimpleNamespace(name="add", arguments='{"a": 1, "b": 1}')
        tc = SimpleNamespace(id="id", function=tc_fn)
        msg.tool_calls = [tc]
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=msg)],
        )

        out = agent.agent_loop(client, user_text="x", model="m", max_turns=1)
        self.assertIn("最大轮次", out)


class TestMain(unittest.TestCase):
    @patch("agent._chat_until_assistant_reply", return_value="ANS")
    @patch("agent.OpenAI")
    @patch("agent.load_env_from_dotenv")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock"})
    def test_main_prints_answer(
        self,
        _ld: MagicMock,
        mock_openai_cls: MagicMock,
        _chat: MagicMock,
    ) -> None:
        mock_openai_cls.return_value = MagicMock()
        with patch.object(sys, "argv", ["agent.py", "hello"]):
            with patch("builtins.input", return_value="/exit"):
                with patch("builtins.print") as p:
                    agent.main()
        printed = [c.args[0] for c in p.call_args_list if c.args]
        self.assertIn("ANS", printed)

    @patch("agent.load_env_from_dotenv")
    def test_main_exits_without_key(self, _ld: MagicMock) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with patch.object(sys, "argv", ["agent.py"]):
                with patch("builtins.print"):
                    with self.assertRaises(SystemExit) as ctx:
                        agent.main()
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
