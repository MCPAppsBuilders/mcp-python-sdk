from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pytest
from pydantic import AnyUrl, FileUrl

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.resources import FileResource, FunctionResource, ResourceManager, ResourceTemplate
from mcp.types import Annotations


@pytest.fixture
def temp_file():
    """Create a temporary file for testing.

    File is automatically cleaned up after the test if it still exists.
    """
    content = "test content"
    with NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(content)
        path = Path(f.name).resolve()
    yield path
    try:  # pragma: no cover
        path.unlink()
    except FileNotFoundError:  # pragma: no cover
        pass  # File was already deleted by the test


class TestResourceManager:
    """Test ResourceManager functionality."""

    def test_add_resource(self, temp_file: Path):
        """Test adding a resource."""
        manager = ResourceManager()
        resource = FileResource(
            uri=FileUrl(f"file://{temp_file}"),
            name="test",
            path=temp_file,
        )
        added = manager.add_resource(resource)
        assert added == resource
        assert manager.list_resources() == [resource]

    def test_add_duplicate_resource(self, temp_file: Path):
        """Test adding the same resource twice."""
        manager = ResourceManager()
        resource = FileResource(
            uri=FileUrl(f"file://{temp_file}"),
            name="test",
            path=temp_file,
        )
        first = manager.add_resource(resource)
        second = manager.add_resource(resource)
        assert first == second
        assert manager.list_resources() == [resource]

    def test_warn_on_duplicate_resources(self, temp_file: Path, caplog: pytest.LogCaptureFixture):
        """Test warning on duplicate resources."""
        manager = ResourceManager()
        resource = FileResource(
            uri=FileUrl(f"file://{temp_file}"),
            name="test",
            path=temp_file,
        )
        manager.add_resource(resource)
        manager.add_resource(resource)
        assert "Resource already exists" in caplog.text

    def test_disable_warn_on_duplicate_resources(self, temp_file: Path, caplog: pytest.LogCaptureFixture):
        """Test disabling warning on duplicate resources."""
        manager = ResourceManager(warn_on_duplicate_resources=False)
        resource = FileResource(
            uri=FileUrl(f"file://{temp_file}"),
            name="test",
            path=temp_file,
        )
        manager.add_resource(resource)
        manager.add_resource(resource)
        assert "Resource already exists" not in caplog.text

    @pytest.mark.anyio
    async def test_get_resource(self, temp_file: Path):
        """Test getting a resource by URI."""
        manager = ResourceManager()
        resource = FileResource(
            uri=FileUrl(f"file://{temp_file}"),
            name="test",
            path=temp_file,
        )
        manager.add_resource(resource)
        retrieved = await manager.get_resource(resource.uri)
        assert retrieved == resource

    @pytest.mark.anyio
    async def test_get_resource_from_template(self):
        """Test getting a resource through a template."""
        manager = ResourceManager()

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        template = ResourceTemplate.from_function(
            fn=greet,
            uri_template="greet://{name}",
            name="greeter",
        )
        manager._templates[template.uri_template] = template

        resource = await manager.get_resource(AnyUrl("greet://world"))
        assert isinstance(resource, FunctionResource)
        content = await resource.read()
        assert content == "Hello, world!"

    @pytest.mark.anyio
    async def test_get_unknown_resource(self):
        """Test getting a non-existent resource."""
        manager = ResourceManager()
        with pytest.raises(ValueError, match="Unknown resource"):
            await manager.get_resource(AnyUrl("unknown://test"))

    def test_list_resources(self, temp_file: Path):
        """Test listing all resources."""
        manager = ResourceManager()
        resource1 = FileResource(
            uri=FileUrl(f"file://{temp_file}"),
            name="test1",
            path=temp_file,
        )
        resource2 = FileResource(
            uri=FileUrl(f"file://{temp_file}2"),
            name="test2",
            path=temp_file,
        )
        manager.add_resource(resource1)
        manager.add_resource(resource2)
        resources = manager.list_resources()
        assert len(resources) == 2
        assert resources == [resource1, resource2]


class TestResourceMetadata:
    """Test resource metadata functionality."""

    def test_add_resource_with_metadata(self, temp_file: Path):
        """Test adding a resource with metadata."""
        metadata = {"version": "1.0", "category": "config"}

        resource = FileResource(
            uri=FileUrl(f"file://{temp_file}"),
            name="test",
            path=temp_file,
            meta=metadata,
        )

        assert resource.meta is not None
        assert resource.meta == metadata
        assert resource.meta["version"] == "1.0"
        assert resource.meta["category"] == "config"

    def test_add_resource_without_metadata(self, temp_file: Path):
        """Test that resources without metadata have None as meta value."""
        resource = FileResource(
            uri=FileUrl(f"file://{temp_file}"),
            name="test",
            path=temp_file,
        )

        assert resource.meta is None

    def test_function_resource_with_metadata(self):
        """Test FunctionResource with metadata via from_function."""

        def get_data() -> str:  # pragma: no cover
            return "test data"

        metadata = {"cache_ttl": 300, "tags": ["data", "readonly"]}

        resource = FunctionResource.from_function(
            fn=get_data,
            uri="resource://data",
            meta=metadata,
        )

        assert resource.meta is not None
        assert resource.meta == metadata
        assert resource.meta["cache_ttl"] == 300
        assert "data" in resource.meta["tags"]

    def test_template_with_metadata(self):
        """Test ResourceTemplate with metadata."""

        def get_user(user_id: str) -> str:  # pragma: no cover
            return f"User {user_id}"

        metadata = {"requires_auth": True, "rate_limit": 100}

        template = ResourceTemplate.from_function(
            fn=get_user,
            uri_template="resource://users/{user_id}",
            meta=metadata,
        )

        assert template.meta is not None
        assert template.meta == metadata
        assert template.meta["requires_auth"] is True
        assert template.meta["rate_limit"] == 100

    @pytest.mark.anyio
    async def test_template_created_resources_inherit_metadata(self):
        """Test that resources created from templates inherit metadata."""

        def get_item(item_id: str) -> str:
            return f"Item {item_id}"

        metadata = {"category": "inventory", "cacheable": True}

        template = ResourceTemplate.from_function(
            fn=get_item,
            uri_template="resource://items/{item_id}",
            meta=metadata,
        )

        # Create a resource from the template
        resource = await template.create_resource("resource://items/123", {"item_id": "123"})

        # The resource should inherit the template's metadata
        assert resource.meta is not None
        assert resource.meta == metadata
        assert resource.meta["category"] == "inventory"

        # Verify the resource works correctly
        content = await resource.read()
        assert content == "Item 123"

    @pytest.mark.anyio
    async def test_metadata_in_fastmcp_decorator(self):
        """Test that metadata is correctly added via FastMCP.resource decorator."""
        app = FastMCP()

        metadata = {"ui": {"component": "file-viewer"}, "priority": "high"}

        @app.resource("resource://config", meta=metadata)
        def get_config() -> str:  # pragma: no cover
            return '{"debug": false}'

        resources = await app.list_resources()
        assert len(resources) == 1
        assert resources[0].meta is not None
        assert resources[0].meta == metadata
        assert resources[0].meta["ui"]["component"] == "file-viewer"
        assert resources[0].meta["priority"] == "high"

    @pytest.mark.anyio
    async def test_metadata_in_template_via_fastmcp_decorator(self):
        """Test that metadata is correctly added to templates via FastMCP.resource decorator."""
        app = FastMCP()

        metadata = {"api_version": "v2", "deprecated": False}

        @app.resource("resource://{city}/weather", meta=metadata)
        def get_weather(city: str) -> str:  # pragma: no cover
            return f"Weather for {city}"

        templates = await app.list_resource_templates()
        assert len(templates) == 1
        assert templates[0].meta is not None
        assert templates[0].meta == metadata
        assert templates[0].meta["api_version"] == "v2"
        assert templates[0].meta["deprecated"] is False

    @pytest.mark.anyio
    async def test_multiple_resources_with_different_metadata(self, temp_file: Path):
        """Test multiple resources with different metadata values."""
        app = FastMCP()

        metadata1 = {"version": "1.0", "public": True}
        metadata2 = {"version": "2.0", "experimental": True}

        @app.resource("resource://v1/data", meta=metadata1)
        def get_v1_data() -> str:  # pragma: no cover
            return "v1 data"

        @app.resource("resource://v2/data", meta=metadata2)
        def get_v2_data() -> str:  # pragma: no cover
            return "v2 data"

        @app.resource("resource://plain")
        def get_plain() -> str:  # pragma: no cover
            return "plain data"

        resources = await app.list_resources()
        assert len(resources) == 3

        # Find resources by URI and check metadata
        resources_by_uri = {str(r.uri): r for r in resources}

        assert resources_by_uri["resource://v1/data"].meta == metadata1
        assert resources_by_uri["resource://v2/data"].meta == metadata2
        assert resources_by_uri["resource://plain"].meta is None

    def test_metadata_with_complex_structure(self):
        """Test metadata with complex nested structures."""

        def complex_resource() -> str:  # pragma: no cover
            return "complex data"

        metadata = {
            "ui": {
                "components": [
                    {"type": "viewer", "options": {"readonly": True}},
                    {"type": "editor", "options": {"syntax": "json"}},
                ],
                "layout": {"position": "sidebar", "width": 300},
            },
            "permissions": ["read", "write"],
            "tags": ["config", "system"],
            "version": 2,
        }

        resource = FunctionResource.from_function(
            fn=complex_resource,
            uri="resource://complex",
            meta=metadata,
        )

        assert resource.meta is not None
        assert resource.meta["ui"]["components"][0]["options"]["readonly"] is True
        assert resource.meta["ui"]["layout"]["width"] == 300
        assert "read" in resource.meta["permissions"]
        assert "config" in resource.meta["tags"]

    def test_metadata_empty_dict(self):
        """Test that empty dict metadata is preserved."""

        def empty_meta_resource() -> str:  # pragma: no cover
            return "data"

        resource = FunctionResource.from_function(
            fn=empty_meta_resource,
            uri="resource://empty-meta",
            meta={},
        )

        assert resource.meta is not None
        assert resource.meta == {}

    @pytest.mark.anyio
    async def test_metadata_with_annotations(self):
        """Test that metadata and annotations can coexist."""
        app = FastMCP()

        metadata = {"custom": "value", "category": "data"}
        annotations = Annotations(priority=0.8)

        @app.resource("resource://combined", meta=metadata, annotations=annotations)
        def combined_resource() -> str:  # pragma: no cover
            return "combined data"

        resources = await app.list_resources()
        assert len(resources) == 1
        assert resources[0].meta == metadata
        assert resources[0].annotations is not None
        assert resources[0].annotations.priority == 0.8

    def test_add_template_with_metadata_via_manager(self):
        """Test adding a template with metadata via ResourceManager."""
        manager = ResourceManager()

        def get_item(id: str) -> str:  # pragma: no cover
            return f"Item {id}"

        metadata = {"source": "database", "cached": True}

        template = manager.add_template(
            fn=get_item,
            uri_template="resource://items/{id}",
            meta=metadata,
        )

        assert template.meta is not None
        assert template.meta == metadata

        # Verify it's in the manager
        templates = manager.list_templates()
        assert len(templates) == 1
        assert templates[0].meta == metadata
