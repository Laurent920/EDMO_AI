import enum
import json
from typing import Awaitable, Callable, Any, TypeVar, Dict
import asyncio


from bleak import BleakClient
from bleak.backends.characteristic import BleakGATTCharacteristic
import requests

from google.protobuf.message import Message as ProtobufMessage

from GoPro import GOPRO_BASE_UUID, GOPRO_BASE_URL, logger, connect_ble

# region ble_command_set_shutter
T = TypeVar("T")
class GoProUuid(str, enum.Enum):
    """UUIDs to write to and receive responses from"""

    COMMAND_REQ_UUID = GOPRO_BASE_UUID.format("0072")
    COMMAND_RSP_UUID = GOPRO_BASE_UUID.format("0073")
    SETTINGS_REQ_UUID = GOPRO_BASE_UUID.format("0074")
    SETTINGS_RSP_UUID = GOPRO_BASE_UUID.format("0075")
    CONTROL_QUERY_SERVICE_UUID = "0000fea6-0000-1000-8000-00805f9b34fb"
    INTERNAL_UUID = "00002a19-0000-1000-8000-00805f9b34fb"
    QUERY_REQ_UUID = GOPRO_BASE_UUID.format("0076")
    QUERY_RSP_UUID = GOPRO_BASE_UUID.format("0077")
    WIFI_AP_SSID_UUID = GOPRO_BASE_UUID.format("0002")
    WIFI_AP_PASSWORD_UUID = GOPRO_BASE_UUID.format("0003")
    NETWORK_MANAGEMENT_REQ_UUID = GOPRO_BASE_UUID.format("0091")
    NETWORK_MANAGEMENT_RSP_UUID = GOPRO_BASE_UUID.format("0092")

    @classmethod
    def dict_by_uuid(cls, value_creator: Callable[["GoProUuid"], T]) -> dict["GoProUuid", T]:
        """Build a dict where the keys are each UUID defined here and the values are built from the input value_creator.

        Args:
            value_creator (Callable[[GoProUuid], T]): callable to create the values from each UUID

        Returns:
            dict[GoProUuid, T]: uuid-to-value mapping.
        """
        return {uuid: value_creator(uuid) for uuid in cls}
    
    
# region ble_command_get_hardware_info
T = TypeVar("T", bound="Response")
class Response:
    """The base class to encapsulate all BLE Responses

    Args:
        uuid (GoProUuid): UUID that this response was received on.
    """

    def __init__(self, uuid: GoProUuid) -> None:
        """Constructor"""
        self.bytes_remaining = 0
        self.uuid = uuid
        self.raw_bytes = bytearray()

    @classmethod
    def from_received_response(cls: type[T], received_response: "Response") -> T:
        """Build a new response from a received response.

        Can be used by subclasses for essentially casting into their derived type.

        Args:
            cls (type[T]): type of response to build
            received_response (Response): received response to build from

        Returns:
            T: built response.
        """
        response = cls(received_response.uuid)
        response.bytes_remaining = 0
        response.raw_bytes = received_response.raw_bytes
        return response

    @property
    def is_received(self) -> bool:
        """Have all of the bytes identified by the length header been received?

        Returns:
            bool: True if received, False otherwise.
        """
        return len(self.raw_bytes) > 0 and self.bytes_remaining == 0

    def accumulate(self, data: bytes) -> None:
        """Accumulate a current packet in to the received response.

        Args:
            data (bytes): bytes to accumulate.
        """
        CONT_MASK = 0b10000000
        HDR_MASK = 0b01100000
        GEN_LEN_MASK = 0b00011111
        EXT_13_BYTE0_MASK = 0b00011111

        class Header(enum.Enum):
            """Header Type Identifiers"""

            GENERAL = 0b00
            EXT_13 = 0b01
            EXT_16 = 0b10
            RESERVED = 0b11

        buf = bytearray(data)
        if buf[0] & CONT_MASK:
            buf.pop(0)
        else:
            # This is a new packet so start with an empty byte array
            self.raw_bytes = bytearray()
            hdr = Header((buf[0] & HDR_MASK) >> 5)
            if hdr is Header.GENERAL:
                self.bytes_remaining = buf[0] & GEN_LEN_MASK
                buf = buf[1:]
            elif hdr is Header.EXT_13:
                self.bytes_remaining = ((buf[0] & EXT_13_BYTE0_MASK) << 8) + buf[1]
                buf = buf[2:]
            elif hdr is Header.EXT_16:
                self.bytes_remaining = (buf[1] << 8) + buf[2]
                buf = buf[3:]

        # Append payload to buffer and update remaining / complete
        self.raw_bytes.extend(buf)
        self.bytes_remaining -= len(buf)
        logger.debug(f"{self.bytes_remaining=}")


class TlvResponse(Response):
    """A Type Length Value TLV Response.

    TLV response all have an ID, status, and payload.
    """

    def __init__(self, uuid: GoProUuid) -> None:
        super().__init__(uuid)
        self.id: int
        self.status: int
        self.payload: bytes

    def parse(self) -> None:
        """Extract the ID, status, and payload"""
        self.id = self.raw_bytes[0]
        self.status = self.raw_bytes[1]
        self.payload = bytes(self.raw_bytes[2:])
        
        
        
# region ble_query_poll_resolution_value 
class Resolution(enum.Enum):
    """Common Resolution Values"""

    RES_4K = 1
    RES_2_7K = 4
    RES_2_7K_4_3 = 6
    RES_1440 = 7
    RES_1080 = 9
    RES_4K_4_3 = 18
    RES_5K = 24


class QueryResponse(TlvResponse):
    """A TLV Response to a Query Operation.

    Args:
        uuid (GoProUuid): _description_
    """

    def __init__(self, uuid: GoProUuid) -> None:
        """Constructor"""
        super().__init__(uuid)
        self.data: dict[int, bytes] = {}

    def parse(self) -> None:
        """Perform common TLV parsing. Then also parse all Query elements into the data property"""
        super().parse()
        buf = bytearray(self.payload)
        while len(buf) > 0:
            # Get ID and Length of query parameter
            param_id = buf[0]
            param_len = buf[1]
            buf = buf[2:]
            # Get the value
            value = buf[:param_len]
            # Store in dict for later access
            self.data[param_id] = bytes(value)

            # Advance the buffer
            buf = buf[param_len:]


# region enable_wifi_ap
async def enable_wifi(identifier: str | None = None) -> tuple[str, str, BleakClient]:
    """Connect to a GoPro via BLE, find its WiFi AP SSID and password, and enable its WiFI AP

    If identifier is None, the first discovered GoPro will be connected to.

    Args:
        identifier (str, optional): Last 4 digits of GoPro serial number. Defaults to None.

    Returns:
        Tuple[str, str]: ssid, password
    """
    # Synchronization event to wait until notification response is received
    event = asyncio.Event()
    client: BleakClient

    async def notification_handler(characteristic: BleakGATTCharacteristic, data: bytearray) -> None:
        uuid = GoProUuid(client.services.characteristics[characteristic.handle].uuid)
        logger.info(f'Received response at {uuid}: {data.hex(":")}')

        # If this is the correct handle and the status is success, the command was a success
        if uuid is GoProUuid.COMMAND_RSP_UUID and data[2] == 0x00:
            logger.info("Command sent successfully")
        # Anything else is unexpected. This shouldn't happen
        else:
            logger.error("Unexpected response")

        # Notify the writer
        event.set()

    client = await connect_ble(notification_handler, identifier)

    # Read from WiFi AP SSID BleUUID
    ssid_uuid = GoProUuid.WIFI_AP_SSID_UUID
    logger.info(f"Reading the WiFi AP SSID at {ssid_uuid}")
    ssid = (await client.read_gatt_char(ssid_uuid.value)).decode()
    logger.info(f"SSID is {ssid}")

    # Read from WiFi AP Password BleUUID
    password_uuid = GoProUuid.WIFI_AP_PASSWORD_UUID
    logger.info(f"Reading the WiFi AP password at {password_uuid}")
    password = (await client.read_gatt_char(password_uuid.value)).decode()
    logger.info(f"Password is {password}")

    # Write to the Command Request BleUUID to enable WiFi
    logger.info("Enabling the WiFi AP")
    event.clear()
    request = bytes([0x03, 0x17, 0x01, 0x01])
    command_request_uuid = GoProUuid.COMMAND_REQ_UUID
    logger.debug(f"Writing to {command_request_uuid}: {request.hex(':')}")
    await client.write_gatt_char(command_request_uuid.value, request, response=True)
    await event.wait()  # Wait to receive the notification response
    logger.info("WiFi AP is enabled")

    return ssid, password, client


# region wifi_command_get_media_list
def get_media_list() -> Dict[str, Any]:
    """Read the media list from the camera and return as JSON

    Returns:
        Dict[str, Any]: complete media list as JSON
    """
    # Build the HTTP GET request
    url = GOPRO_BASE_URL + "/gopro/media/list"
    logger.info(f"Getting the media list: sending {url}")

    # Send the GET request and retrieve the response
    response = requests.get(url, timeout=10)
    # Check for errors (if an error is found, an exception will be raised)
    response.raise_for_status()
    logger.info("Command sent successfully")
    # Log response as json
    logger.info(f"Response: {json.dumps(response.json(), indent=4)}")

    return response.json()


# region set_turbo_mode
class ProtobufResponse(Response):
    """Accumulate and parse protobuf responses"""

    def __init__(self, uuid: GoProUuid) -> None:
        super().__init__(uuid)
        self.feature_id: int
        self.action_id: int
        self.uuid = uuid
        self.data: ProtobufMessage

    def parse(self, proto_message: type[ProtobufMessage]) -> None:
        """Set the responses data by parsing using the passed in protobuf container

        Args:
            proto_message (type[ProtobufMessage]): protobuf container to use for parsing
        """
        self.feature_id = self.raw_bytes[0]
        self.action_id = self.raw_bytes[1]
        self.data = proto_message.FromString(bytes(self.raw_bytes[2:]))
