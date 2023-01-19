import numpy as np
import cv2, base64


def fuse_images(
    images,
    image_size=None,
    row=0,
    col=0,
    is_row_major=True,
    is_portrait=False,
    row_spacing=0,
    col_spacing=0,
    border_left=0,
    border_right=0,
    border_top=0,
    border_bottom=0,
    black_background=True,
):
    """Fuses a collection of images into an entire image.

    Args:
      images: A collection of images to fuse. Should be with shape [num, height,
        width, channels].
      image_size: This field is used to resize the image before fusion. `0`
        disables resizing. (default: None)
      row: Number of rows used for image fusion. If not set, this field will be
        automatically assigned based on `col` and total number of images.
        (default: None)
      col: Number of columns used for image fusion. If not set, this field will be
        automatically assigned based on `row` and total number of images.
        (default: None)
      is_row_major: Whether the input images should be arranged row-major or
        column-major. (default: True)
      is_portrait: Only active when both `row` and `col` should be assigned
        automatically. (default: False)
      row_spacing: Space between rows. (default: 0)
      col_spacing: Space between columns. (default: 0)
      border_left: Width of left border. (default: 0)
      border_right: Width of right border. (default: 0)
      border_top: Width of top border. (default: 0)
      border_bottom: Width of bottom border. (default: 0)

    Returns:
      The fused image.

    Raises:
      ValueError: If the input `images` is not with shape [num, height, width,
        width].
    """
    if images is None:
        return images

    if images.ndim != 4:
        raise ValueError(
            f"Input `images` should be with shape [num, height, "
            f"width, channels], but {images.shape} is received!"
        )

    num, image_height, image_width, channels = images.shape
    width, height = parse_image_size(image_size)
    height = height or image_height
    width = width or image_width
    row, col = get_grid_shape(num, row=row, col=col, is_portrait=is_portrait)
    fused_height = height * row + row_spacing * (row - 1) + border_top + border_bottom
    fused_width = width * col + col_spacing * (col - 1) + border_left + border_right
    fused_image = get_blank_image(
        fused_height, fused_width, channels=channels, is_black=black_background
    )
    images = images.reshape(row, col, image_height, image_width, channels)
    if not is_row_major:
        images = images.transpose(1, 0, 2, 3, 4)

    for i in range(row):
        y = border_top + i * (height + row_spacing)
        for j in range(col):
            x = border_left + j * (width + col_spacing)
            if height != image_height or width != image_width:
                image = cv2.resize(images[i, j], (width, height))
            else:
                image = images[i, j]
            fused_image[y : y + height, x : x + width] = image

    return fused_image


def get_sortable_html_header(column_name_list, sort_by_ascending=False):
    """Gets header for sortable html page.

    Basically, the html page contains a sortable table, where user can sort the
    rows by a particular column by clicking the column head.

    Example:

    column_name_list = [name_1, name_2, name_3]
    header = get_sortable_html_header(column_name_list)
    footer = get_sortable_html_footer()
    sortable_table = ...
    html_page = header + sortable_table + footer

    Args:
      column_name_list: List of column header names.
      sort_by_ascending: Default sorting order. If set as `True`, the html page
        will be sorted by ascending order when the header is clicked for the first
        time.

    Returns:
      A string, which represents for the header for a sortable html page.
    """
    header = "\n".join(
        [
            '<script type="text/javascript">',
            "var column_idx;",
            "var sort_by_ascending = " + str(sort_by_ascending).lower() + ";",
            "",
            "function sorting(tbody, column_idx){",
            "  this.column_idx = column_idx;",
            "  Array.from(tbody.rows)",
            "       .sort(compareCells)",
            "       .forEach(function(row) { tbody.appendChild(row); })",
            "  sort_by_ascending = !sort_by_ascending;",
            "}",
            "",
            "function compareCells(row_a, row_b) {",
            "  var val_a = row_a.cells[column_idx].innerText;",
            "  var val_b = row_b.cells[column_idx].innerText;",
            "  var flag = sort_by_ascending ? 1 : -1;",
            "  return flag * (val_a > val_b ? 1 : -1);",
            "}",
            "</script>",
            "",
            "<html>",
            "",
            "<head>",
            "<style>",
            "  table {",
            "    border-spacing: 0;",
            "    border: 1px solid black;",
            "  }",
            "  th {",
            "    cursor: pointer;",
            "  }",
            "  th, td {",
            "    text-align: left;",
            "    vertical-align: middle;",
            "    border-collapse: collapse;",
            "    border: 0.5px solid black;",
            "    padding: 8px;",
            "  }",
            "  tr:nth-child(even) {",
            "    background-color: #d2d2d2;",
            "  }",
            "</style>",
            "</head>",
            "",
            "<body>",
            "",
            "<table>",
            "<thead>",
            "<tr>",
            "",
        ]
    )
    for idx, column_name in enumerate(column_name_list):
        header += f'  <th onclick="sorting(tbody, {idx})">{column_name}</th>\n'
    header += "</tr>\n"
    header += "</thead>\n"
    header += '<tbody id="tbody">\n'

    return header


def get_sortable_html_footer():
    """Gets footer for sortable html page.

    Check function `get_sortable_html_header()` for more details.
    """
    return "</tbody>\n</table>\n\n</body>\n</html>\n"


def encode_image_to_html_str(image, image_size=None):
    """Encodes an image to html language.

    NOTE: Input image is always assumed to be with `RGB` channel order.

    Args:
      image: The input image to encode. Should be with `RGB` channel order.
      image_size: This field is used to resize the image before encoding. `0`
        disables resizing. (default: None)

    Returns:
      A string which represents the encoded image.
    """
    if image is None:
        return ""

    assert image.ndim == 3 and image.shape[2] in [1, 3]

    # Change channel order to `BGR`, which is opencv-friendly.
    image = image[:, :, ::-1]

    # Resize the image if needed.
    width, height = parse_image_size(image_size)
    if height or width:
        height = height or image.shape[0]
        width = width or image.shape[1]
        image = cv2.resize(image, (width, height))

    # Encode the image to html-format string.
    encoded_image = cv2.imencode(".jpg", image)[1].tostring()
    encoded_image_base64 = base64.b64encode(encoded_image).decode("utf-8")
    html_str = f'<img src="data:image/jpeg;base64, {encoded_image_base64}"/>'

    return html_str


def decode_html_str_to_image(html_str, image_size=None):
    """Decodes image from html.

    Args:
      html_str: Image string parsed from html.
      image_size: This field is used to resize the image after decoding. `0`
        disables resizing. (default: None)

    Returns:
      An image with `RGB` channel order.
    """
    if not html_str:
        return None

    assert isinstance(html_str, str)
    image_str = html_str.split(",")[-1]
    encoded_image = base64.b64decode(image_str)
    encoded_image_numpy = np.frombuffer(encoded_image, dtype=np.uint8)
    image = cv2.imdecode(encoded_image_numpy, flags=cv2.IMREAD_COLOR)

    # Resize the image if needed.
    width, height = parse_image_size(image_size)
    if height or width:
        height = height or image.shape[0]
        width = width or image.shape[1]
        image = cv2.resize(image, (width, height))

    return image[:, :, ::-1]


def get_grid_shape(size, row=0, col=0, is_portrait=False):
    """Gets the shape of a grid based on the size.

    This function makes greatest effort on making the output grid square if
    neither `row` nor `col` is set. If `is_portrait` is set as `False`, the height
    will always be equal to or smaller than the width. For example, if input
    `size = 16`, output shape will be `(4, 4)`; if input `size = 15`, output shape
    will be (3, 5). Otherwise, the height will always be equal to or larger than
    the width.

    Args:
      size: Size (height * width) of the target grid.
      is_portrait: Whether to return a portrait size of a landscape size.
        (default: False)

    Returns:
      A two-element tuple, representing height and width respectively.
    """
    assert isinstance(size, int)
    assert isinstance(row, int)
    assert isinstance(col, int)
    if size == 0:
        return (0, 0)

    if row > 0 and col > 0 and row * col != size:
        row = 0
        col = 0

    if row > 0 and size % row == 0:
        return (row, size // row)
    if col > 0 and size % col == 0:
        return (size // col, col)

    row = int(np.sqrt(size))
    while row > 0:
        if size % row == 0:
            col = size // row
            break
        row = row - 1

    return (col, row) if is_portrait else (row, col)


def get_blank_image(height, width, channels=3, is_black=True):
    """Gets a blank image, either white of black.

    NOTE: This function will always return an image with `RGB` channel order for
    color image and pixel range [0, 255].

    Args:
      height: Height of the returned image.
      width: Width of the returned image.
      channels: Number of channels. (default: 3)
      is_black: Whether to return a black image or white image. (default: True)
    """
    shape = (height, width, channels)
    if is_black:
        return np.zeros(shape, dtype=np.uint8)
    return np.ones(shape, dtype=np.uint8) * 255


def add_text_to_image(
    image,
    text="",
    position=None,
    font=cv2.FONT_HERSHEY_TRIPLEX,
    font_size=1.0,
    line_type=cv2.LINE_8,
    line_width=1,
    color=(255, 255, 255),
):
    """Overlays text on given image.

    NOTE: The input image is assumed to be with `RGB` channel order.

    Args:
      image: The image to overlay text on.
      text: Text content to overlay on the image. (default: '')
      position: Target position (bottom-left corner) to add text. If not set,
        center of the image will be used by default. (default: None)
      font: Font of the text added. (default: cv2.FONT_HERSHEY_TRIPLEX)
      font_size: Font size of the text added. (default: 1.0)
      line_type: Line type used to depict the text. (default: cv2.LINE_8)
      line_width: Line width used to depict the text. (default: 1)
      color: Color of the text added in `RGB` channel order. (default:
        (255, 255, 255))

    Returns:
      An image with target text overlayed on.
    """
    if image is None or not text:
        return image

    cv2.putText(
        img=image,
        text=text,
        org=position,
        fontFace=font,
        fontScale=font_size,
        color=color,
        thickness=line_width,
        lineType=line_type,
        bottomLeftOrigin=False,
    )

    return image


def parse_image_size(obj):
    """Parses object to a pair of image size, i.e., (width, height).

    Args:
      obj: The input object to parse image size from.

    Returns:
      A two-element tuple, indicating image width and height respectively.

    Raises:
      If the input is invalid, i.e., neither a list or tuple, nor a string.
    """
    if obj is None or obj == "":
        width = height = 0
    elif isinstance(obj, int):
        width = height = obj
    elif isinstance(obj, (list, tuple, np.ndarray)):
        numbers = tuple(obj)
        if len(numbers) == 0:
            width = height = 0
        elif len(numbers) == 1:
            width = height = numbers[0]
        elif len(numbers) == 2:
            width = numbers[0]
            height = numbers[1]
        else:
            raise ValueError(f"At most two elements for image size.")
    elif isinstance(obj, str):
        splits = obj.replace(" ", "").split(",")
        numbers = tuple(map(int, splits))
        if len(numbers) == 0:
            width = height = 0
        elif len(numbers) == 1:
            width = height = numbers[0]
        elif len(numbers) == 2:
            width = numbers[0]
            height = numbers[1]
        else:
            raise ValueError(f"At most two elements for image size.")
    else:
        raise ValueError(f"Invalid type of input: {type(obj)}!")

    return (max(0, width), max(0, height))




class HtmlPageVisualizer(object):
    """Defines the html page visualizer.

    This class can be used to visualize image results as html page. Basically, it
    is based on an html-format sorted table with helper functions
    `get_sortable_html_header()`, `get_sortable_html_footer()`, and
    `encode_image_to_html_str()`. To simplify the usage, specifying the following
    fields is enough to create a visualization page:

    (1) num_rows: Number of rows of the table (header-row exclusive).
    (2) num_cols: Number of columns of the table.
    (3) header contents (optional): Title of each column.

    NOTE: `grid_size` can be used to assign `num_rows` and `num_cols`
    automatically.

    Example:

    html = HtmlPageVisualizer(num_rows, num_cols)
    html.set_headers([...])
    for i in range(num_rows):
      for j in range(num_cols):
        html.set_cell(i, j, text=..., image=..., highlight=False)
    html.save('visualize.html')
    """

    def __init__(
        self, num_rows=0, num_cols=0, grid_size=0, is_portrait=False, viz_size=None
    ):
        if grid_size > 0:
            num_rows, num_cols = get_grid_shape(
                grid_size, row=num_rows, col=num_cols, is_portrait=is_portrait
            )
        assert num_rows > 0 and num_cols > 0

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.viz_size = parse_image_size(viz_size)
        self.headers = ["" for _ in range(self.num_cols)]
        self.cells = [
            [
                {
                    "text": "",
                    "image": "",
                    "highlight": False,
                }
                for _ in range(self.num_cols)
            ]
            for _ in range(self.num_rows)
        ]

    def set_header(self, col_idx, content):
        """Sets the content of a particular header by column index."""
        self.headers[col_idx] = content

    def set_headers(self, contents):
        """Sets the contents of all headers."""
        if isinstance(contents, str):
            contents = [contents]
        assert isinstance(contents, (list, tuple))
        assert len(contents) == self.num_cols
        for col_idx, content in enumerate(contents):
            self.set_header(col_idx, content)

    def set_cell(self, row_idx, col_idx, text="", image=None, highlight=False):
        """Sets the content of a particular cell.

        Basically, a cell contains some text as well as an image. Both text and
        image can be empty.

        Args:
          row_idx: Row index of the cell to edit.
          col_idx: Column index of the cell to edit.
          text: Text to add into the target cell. (default: None)
          image: Image to show in the target cell. Should be with `RGB` channel
            order. (default: None)
          highlight: Whether to highlight this cell in the html. (default: False)
        """
        self.cells[row_idx][col_idx]["text"] = text
        self.cells[row_idx][col_idx]["image"] = encode_image_to_html_str(
            image, self.viz_size
        )
        self.cells[row_idx][col_idx]["highlight"] = bool(highlight)

    def sequential_set_cell(self, images):
        """
        Sequentially assign the image to the cell
        """
        for i in range(len(images)):
            x, y = i // self.num_cols, i % self.num_cols
            self.set_cell(x, y, image=images[i])

    def save(self, save_path):
        """Saves the html page."""
        html = ""
        for i in range(self.num_rows):
            html += f"<tr>\n"
            for j in range(self.num_cols):
                text = self.cells[i][j]["text"]
                image = self.cells[i][j]["image"]
                color = ' bgcolor="#FF8888"' if self.cells[i][j]["highlight"] else ""
                if text:
                    html += f"  <td{color}>{text}<br><br>{image}</td>\n"
                else:
                    html += f"  <td{color}>{image}</td>\n"
            html += f"</tr>\n"

        header = get_sortable_html_header(self.headers)
        footer = get_sortable_html_footer()

        with open(save_path, "w") as f:
            f.write(header + html + footer)

