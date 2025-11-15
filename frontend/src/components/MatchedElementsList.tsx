import { MatchedElement } from "../types";

interface MatchedElementsListProps {
  elements: MatchedElement[];
  onHighlight: (elementId: string, pageNumber: number) => void;
  highlightedElementId: string | null;
}

const MatchedElementsList = ({
  elements,
  onHighlight,
  highlightedElementId
}: MatchedElementsListProps) => {
  if (!elements.length) {
    return <div className="card">No matched elements.</div>;
  }

  return (
    <div className="card" style={{ overflowX: "auto" }}>
      <table className="matched-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Type</th>
            <th>Page</th>
            <th>Snippet</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {elements.map((element) => (
            <tr key={element.element_id}>
              <td>{element.element_id}</td>
              <td>{element.type}</td>
              <td>{element.page_number}</td>
              <td>
                {element.content.length > 120
                  ? `${element.content.slice(0, 117)}…`
                  : element.content}
              </td>
              <td>
                <button
                  className="btn btn--secondary"
                  onClick={() =>
                    onHighlight(element.element_id, element.page_number)
                  }
                >
                  {highlightedElementId === element.element_id
                    ? "Highlighted"
                    : "View"}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default MatchedElementsList;

