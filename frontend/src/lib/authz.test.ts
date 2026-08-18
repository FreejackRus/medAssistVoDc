import { describe, expect, it } from "vitest";
import { canManageUsers } from "@/lib/authz";

describe("canManageUsers", () => {
  it("allows admin and manager roles only", () => {
    expect(canManageUsers("admin")).toBe(true);
    expect(canManageUsers("manager")).toBe(true);
    expect(canManageUsers("user")).toBe(false);
    expect(canManageUsers(undefined)).toBe(false);
  });
});
