import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Select } from "@/components/ui/select";

describe("Select", () => {
  it("uses native keyboard selection semantics", async () => {
    const onValueChange = vi.fn();
    render(
      <Select
        id="role"
        value="user"
        options={[
          { value: "user", label: "Пользователь" },
          { value: "manager", label: "Менеджер" },
        ]}
        onValueChange={onValueChange}
      />,
    );

    const select = screen.getByRole("combobox");
    await userEvent.selectOptions(select, "manager");

    expect(onValueChange).toHaveBeenCalledWith("manager");
  });
});
