import { useState } from "react";

interface JSONTreeProps {
  data: unknown;
  label?: string;
}

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const JSONTree = ({ data, label = "root" }: JSONTreeProps) => {
  const [open, setOpen] = useState(true);

  if (!isObject(data) && !Array.isArray(data)) {
    return (
      <div className="json-node">
        <span className="json-key">{label}: </span>
        <span className="json-value">{JSON.stringify(data)}</span>
      </div>
    );
  }

  const entries = Array.isArray(data)
    ? data.map((item, index) => [index, item])
    : Object.entries(data);

  return (
    <div className="json-node">
      <div
        className="json-key"
        onClick={() => setOpen((prev) => !prev)}
        role="button"
      >
        {open ? "▼" : "▶"} {label}
      </div>
      {open &&
        entries.map(([key, value]) => (
          <JSONTree key={String(key)} data={value} label={String(key)} />
        ))}
    </div>
  );
};

export default JSONTree;

