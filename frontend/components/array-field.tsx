"use client";
import * as React from "react";
import { useArrayField } from "@wandry/inertia-form";

import {
  Field,
  FieldDescription,
  FieldError,
  FieldLabel,
} from "@/components/ui/field";
import { Button } from "@/components/ui/button";

export type ArrayFieldItemProps = {
  append: (defaultValue?: any) => void;
  remove: () => void;
  move: (from: number, to: number) => void;
  canAppend: () => boolean;
  canRemove: (index: number) => boolean;
  canMove: (from: number, to: number) => boolean;
};

export type ArrayFieldProps = {
  name: string;
  children: (name: string, props: ArrayFieldItemProps) => React.ReactNode;
  addButton?: React.ReactNode;
  label?: string;
  description?: string;
};

const ArrayField: React.FC<ArrayFieldProps> = ({
  name,
  label,
  description,
  addButton,
  children,
}) => {
  const { value, error, ...field } = useArrayField(name, { defaultValue: [] });

  return (
    <Field>
      {label && <FieldLabel>{label}</FieldLabel>}
      {description && <FieldDescription>{description}</FieldDescription>}
      {value.map((_, index) =>
        children(`${name}[${index}]`, {
          ...field,
          remove: () => field.remove(index),
          canRemove: () => field.canRemove(index),
          move: (to: number) => field.move(index, to),
        })
      )}
      {addButton ?? (
        <Button type="button" onClick={() => field.append()}>
          Add
        </Button>
      )}
      {error && <FieldError>{error}</FieldError>}
    </Field>
  );
};

export default ArrayField;
